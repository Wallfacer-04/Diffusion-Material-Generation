"""周期性近邻特征（minimum-image convention）。

这是相对 #12 的关键升级之一：网络不再把结构当成一条向量，
而是通过「最小镜像」计算任意两个原子在周期晶格下的真实距离，
让模型能感知到「跨周期边界其实相邻」这件事。
"""

from __future__ import annotations

import torch


def periodic_edge_rbf(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    centers: torch.Tensor,
    width: float,
) -> torch.Tensor:
    """对所有原子对计算最小镜像距离的高斯 RBF 特征。

    frac_coords: (B, N, 3) 分数坐标
    lattice:     (B, 3, 3) 每一行是晶格向量
    centers:     (F,) RBF 中心（埃）
    返回:        (B, N, N, F)
    """
    delta = frac_coords[:, :, None, :] - frac_coords[:, None, :, :]  # (B, N, N, 3)
    delta = delta - torch.round(delta)                               # wrap 到 [-0.5, 0.5)
    cart = delta @ lattice.unsqueeze(1)                              # (B, N, N, 3)，笛卡尔位移
    dist = cart.norm(dim=-1)                                         # (B, N, N)
    return torch.exp(-((dist[..., None] - centers) ** 2) / (width ** 2))
