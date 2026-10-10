"""周期性近邻特征（用 Gram 矩阵算距离，天生旋转不变）。

最小镜像的分数位移 Δx_ij = wrap(x_i - x_j)，它本身就是旋转不变量
（旋转作用在晶格上，不改变分数坐标）。两原子的笛卡尔距离平方为

    d^2_ij = Δx_ij · G · Δx_ij^T,   G = L·L^T

注意这里直接用 Gram 而不是晶格 L，所以整条边特征只依赖旋转不变量。
"""

from __future__ import annotations

import torch


def gram6_to_matrix(gram6: torch.Tensor) -> torch.Tensor:
    """(B, 6) → (B, 3, 3) 对称矩阵。"""
    batch = gram6.shape[0]
    matrix = gram6.new_zeros(batch, 3, 3)
    matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 2, 2] = gram6[:, 0], gram6[:, 1], gram6[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = gram6[:, 3]
    matrix[:, 0, 2] = matrix[:, 2, 0] = gram6[:, 4]
    matrix[:, 1, 2] = matrix[:, 2, 1] = gram6[:, 5]
    return matrix


def periodic_edges(frac_coords: torch.Tensor, gram6: torch.Tensor, centers: torch.Tensor, width: float):
    """返回 (rbf, delta, dist)。

    frac_coords: (B, N, 3) 分数坐标
    gram6:       (B, 6) Gram 分量
    rbf:         (B, N, N, F) 距离的高斯 RBF
    delta:       (B, N, N, 3) 最小镜像分数位移（相对位移，供 EGNN 式更新用）
    dist:        (B, N, N) 距离
    """
    matrix = gram6_to_matrix(gram6)                                  # (B, 3, 3)
    delta = frac_coords[:, :, None, :] - frac_coords[:, None, :, :]  # (B, N, N, 3)
    delta = delta - torch.round(delta)                               # 最小镜像，wrap 到 [-0.5, 0.5)
    tmp = torch.einsum("bnmc,bcd->bnmd", delta, matrix)              # Δx · G
    dist2 = (tmp * delta).sum(dim=-1).clamp(min=1e-8)                # Δx · G · Δx^T
    dist = dist2.sqrt()
    rbf = torch.exp(-((dist[..., None] - centers) ** 2) / (width ** 2))
    return rbf, delta, dist
