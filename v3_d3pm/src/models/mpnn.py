"""消息传递 GNN 去噪器（v2）。

沿用 v1 已验证有效的骨干：周期近邻图 + 消息传递 + 槽位嵌入 +
时间嵌入 + 全局晶格特征；三个输出头分别对应三部分：

  - species_head：输出 (K) 个 **logits**，用于 D3PM 预测干净类别 x_0
  - coord_head  ：输出 (3) 个坐标噪声
  - lattice_head：输出 (9) 个晶格噪声（由全局读出得到）
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..data.graph import periodic_edge_rbf


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.float()[:, None] * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, h: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        n = h.shape[1]
        hi = h.unsqueeze(2).expand(-1, -1, n, -1)
        hj = h.unsqueeze(1).expand(-1, n, -1, -1)
        msg = self.message(torch.cat([hi, hj, edge], dim=-1)).sum(dim=2)
        return h + self.update(torch.cat([h, msg], dim=-1))


class CrystalDenoiser(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_elements: int,
        num_species_input: int | None = None,
        hidden: int = 64,
        num_layers: int = 3,
        rbf_dim: int = 16,
        r_max: float = 8.0,
        time_dim: int = 32,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_elements = num_elements
        self.time_dim = time_dim
        # mask 型扩散多一个 [MASK] 类别，所以词表大小可单独指定
        self.num_species_input = num_species_input if num_species_input is not None else num_elements
        # 原子种类用查表嵌入（与 GemNet 的 AtomEmbedding 一致），而不是 one-hot + Linear。
        # 两者数学上等价；查表省掉 K×hidden 次无谓乘加，也不额外引入 bias。
        self.species_embedding = nn.Embedding(self.num_species_input, hidden)
        # 初始化用 1/sqrt(词表大小) 的均匀分布：和原先 one-hot + Linear 的等效尺度一致。
        # （GemNet 用 uniform(-√3, √3) 即单位方差；实测在这个小任务上偏大、收敛更差，
        #   见 README「初始化的取舍」一节。）
        bound = 1.0 / math.sqrt(self.num_species_input)
        nn.init.uniform_(self.species_embedding.weight, a=-bound, b=bound)
        self.coord_proj = nn.Linear(3, hidden)
        self.lattice_proj = nn.Linear(9, hidden)
        self.index_embedding = nn.Parameter(torch.randn(num_atoms, hidden) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.register_buffer("rbf_centers", torch.linspace(0.0, r_max, rbf_dim))
        self.rbf_width = r_max / rbf_dim
        self.layers = nn.ModuleList(MessagePassingLayer(hidden, rbf_dim) for _ in range(num_layers))
        self.species_head = nn.Linear(hidden, num_elements)   # logits，不是噪声
        self.coord_head = nn.Linear(hidden, 3)
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9)
        )

    def forward(
        self,
        species_index: torch.Tensor,
        coords: torch.Tensor,
        lattice: torch.Tensor,
        t: torch.Tensor,
    ):
        h = self.species_embedding(species_index) + self.coord_proj(coords) + self.index_embedding.unsqueeze(0)
        h = h + self.time_mlp(sinusoidal(t, self.time_dim)).unsqueeze(1)
        h = h + self.lattice_proj(lattice.reshape(-1, 9)).unsqueeze(1)

        edge = periodic_edge_rbf(coords, lattice, self.rbf_centers, self.rbf_width)
        for layer in self.layers:
            h = layer(h, edge)

        species_logits = self.species_head(h)
        coord_noise = self.coord_head(h)
        lattice_noise = self.lattice_head(h.mean(dim=1)).view(-1, 3, 3)
        return species_logits, coord_noise, lattice_noise
