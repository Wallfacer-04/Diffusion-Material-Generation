"""EGNN 风格的等变去噪网络（v2）。

与 v1 相比的三处关键改动：

1. **旋转不变**：晶格只通过 Gram 矩阵 G = L·L^T 进入网络，G 对旋转不变，
   所以整个网络对"晶体朝向"不敏感。
2. **置换对称是可选的**（`use_slot` 开关）：
   - `use_slot=False`：不加槽位嵌入，节点特征只由「自身物种 + 自身坐标 + 时间 + 全局 Gram」
     构成，消息用 `sum` 聚合 → 严格置换等变（原子重排，输出同步重排）。
   - `use_slot=True`（默认）：每个格位有独立嵌入 → 打破置换对称。
   为什么默认打开：本任务是"背下一个结构"，本身就依赖"哪个格位是哪个原子"，
   关掉槽位后重建会明显变差（README 里有实测对比）。
3. **相对位移**：消息建立在最小镜像的**相对位移** Δx_ij 上；并额外用
   `Σ_j w_ij·Δx_ij` 这种「不变量 × 相对向量」的形式给出一个向量场
   （EGNN 的做法），由一个可学习门控决定使用多少。

注意：节点特征里含绝对分数坐标，因此模型对「周期原点的整体平移」这一规范自由
不是不变的（单结构任务下无影响）。这是刻意保留的简化。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..data.graph import periodic_edges


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.float()[:, None] * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class EGNNLayer(nn.Module):
    """一层消息传递 + 相对位移向量场。"""

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        # 把每条边的消息压成一个标量权重，乘到相对位移上
        self.coord_weight = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, edge: torch.Tensor, delta: torch.Tensor):
        n = h.shape[1]
        hi = h.unsqueeze(2).expand(-1, -1, n, -1)
        hj = h.unsqueeze(1).expand(-1, n, -1, -1)
        msg = self.message(torch.cat([hi, hj, edge], dim=-1))           # (B, N, N, H)

        h = h + self.update(torch.cat([h, msg.sum(dim=2)], dim=-1))      # 逐节点更新

        weight = self.coord_weight(msg)                                  # (B, N, N, 1)
        vector = (weight * delta).sum(dim=2)                             # (B, N, 3)
        return h, vector


class CrystalEGNN(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_elements: int,
        hidden: int = 64,
        num_layers: int = 3,
        rbf_dim: int = 16,
        r_max: float = 8.0,
        time_dim: int = 32,
        use_slot: bool = True,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.time_dim = time_dim
        self.use_slot = use_slot
        self.species_proj = nn.Linear(num_elements, hidden)
        self.coord_proj = nn.Linear(3, hidden)
        self.gram_proj = nn.Linear(6, hidden)
        if use_slot:
            # 每个格位的独立嵌入：打破置换对称，让模型能记住"第 i 个原子是谁"
            self.index_embedding = nn.Parameter(torch.randn(num_atoms, hidden) * 0.02)
        else:
            self.index_embedding = None
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.register_buffer("rbf_centers", torch.linspace(0.0, r_max, rbf_dim))
        self.rbf_width = r_max / rbf_dim
        self.layers = nn.ModuleList(EGNNLayer(hidden, rbf_dim) for _ in range(num_layers))
        self.species_head = nn.Linear(hidden, num_elements)
        self.coord_head = nn.Linear(hidden, 3)
        self.gram_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 6))
        # 相对位移向量场的门控，初始为 0（先不用，学会用再打开）
        self.coord_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        species: torch.Tensor,
        coords: torch.Tensor,
        gram6: torch.Tensor,
        t: torch.Tensor,
    ):
        edge, delta, _ = periodic_edges(coords, gram6, self.rbf_centers, self.rbf_width)

        h = self.species_proj(species) + self.coord_proj(coords)
        if self.index_embedding is not None:
            h = h + self.index_embedding.unsqueeze(0)
        h = h + self.time_mlp(sinusoidal(t, self.time_dim)).unsqueeze(1)
        h = h + self.gram_proj(gram6).unsqueeze(1)

        vector_total = torch.zeros_like(coords)
        for layer in self.layers:
            h, vector = layer(h, edge, delta)
            vector_total = vector_total + vector

        species_noise = self.species_head(h)
        coord_noise = self.coord_head(h) + self.coord_gate * vector_total
        gram_noise = self.gram_head(h.mean(dim=1))
        return species_noise, coord_noise, gram_noise
