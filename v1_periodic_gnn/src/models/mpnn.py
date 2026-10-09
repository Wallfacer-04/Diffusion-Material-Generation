"""消息传递 GNN 去噪器（置换不变 + 周期近邻）。
骨干为图神经网络：
  - 节点：每个原子，特征是 one-hot 物种 + 槽位嵌入 + 时间嵌入
  - 边：所有原子对，特征是最小镜像距离的高斯 RBF
  - 层：消息 m_ij = MLP([h_i, h_j, e_ij])，聚合 sum_j，再更新 h_i

聚合用求和 → 对原子排列置换不变（换顺序结果一样），这是主流晶体模型的核心性质。
输出头同时预测三部分噪声：物种 (K)、坐标 (3)、以及由全局读出得到的晶格 (3x3)。
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
        hidden: int = 64,
        num_layers: int = 3,
        rbf_dim: int = 16,
        r_max: float = 8.0,
        time_dim: int = 32,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.time_dim = time_dim
        self.species_proj = nn.Linear(num_elements, hidden)
        self.coord_proj = nn.Linear(3, hidden)
        self.lattice_proj = nn.Linear(9, hidden)
        self.index_embedding = nn.Parameter(torch.randn(num_atoms, hidden) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.register_buffer("rbf_centers", torch.linspace(0.0, r_max, rbf_dim))
        self.rbf_width = r_max / rbf_dim
        self.layers = nn.ModuleList(MessagePassingLayer(hidden, rbf_dim) for _ in range(num_layers))
        self.species_head = nn.Linear(hidden, num_elements)
        self.coord_head = nn.Linear(hidden, 3)
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9)
        )

    def forward(
        self,
        species: torch.Tensor,
        coords: torch.Tensor,
        lattice: torch.Tensor,
        t: torch.Tensor,
    ):
        # 三部分都要直接进入节点特征，否则坐标/晶格的噪声无法从距离里还原
        h = self.species_proj(species) + self.coord_proj(coords) + self.index_embedding.unsqueeze(0)
        h = h + self.time_mlp(sinusoidal(t, self.time_dim)).unsqueeze(1)
        h = h + self.lattice_proj(lattice.reshape(-1, 9)).unsqueeze(1)

        edge = periodic_edge_rbf(coords, lattice, self.rbf_centers, self.rbf_width)
        for layer in self.layers:
            h = layer(h, edge)

        species_noise = self.species_head(h)
        coord_noise = self.coord_head(h)
        lattice_noise = self.lattice_head(h.mean(dim=1)).view(-1, 3, 3)
        return species_noise, coord_noise, lattice_noise
