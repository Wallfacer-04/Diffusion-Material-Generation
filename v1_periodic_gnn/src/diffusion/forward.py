"""前向加噪与训练目标（epsilon-prediction）。

物种、坐标、晶格三部分用同一套调度，各自独立加噪。
物种走的是「one-hot 连续松弛」：先在 one-hot 上做高斯加噪，
采样时再取 argmax 变回离散元素。MatterGen 对物种用的是 D3PM 离散扩散，
这里用连续松弛作为更易上手的简化（README 里有说明）。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .schedule import CosineSchedule


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: CosineSchedule, noise: torch.Tensor | None = None) -> torch.Tensor:
    """从 q(x_t | x_0) 直接采样带噪张量。"""
    if noise is None:
        noise = torch.randn_like(x0)
    s = schedule
    return s.gather(s.sqrt_alpha_bar, t, x0) * x0 + s.gather(s.sqrt_one_minus_alpha_bar, t, x0) * noise


def training_loss(
    model: nn.Module,
    schedule: CosineSchedule,
    species0: torch.Tensor,
    coords0: torch.Tensor,
    lattice0: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """联合噪声预测损失，返回 (总损失, 各部分损失明细)。"""
    batch = species0.shape[0]
    device = species0.device
    t = torch.randint(schedule.num_steps, (batch,), device=device)

    noise_s = torch.randn_like(species0)
    noise_c = torch.randn_like(coords0)
    noise_l = torch.randn_like(lattice0)

    species_t = q_sample(species0, t, schedule, noise_s)
    coords_t = q_sample(coords0, t, schedule, noise_c)
    lattice_t = q_sample(lattice0, t, schedule, noise_l)

    pred_s, pred_c, pred_l = model(species_t, coords_t, lattice_t, t)

    ws, wc, wl = weights
    loss_s = F.mse_loss(pred_s, noise_s)
    loss_c = F.mse_loss(pred_c, noise_c)
    loss_l = F.mse_loss(pred_l, noise_l)
    total = ws * loss_s + wc * loss_c + wl * loss_l
    parts = {"species": loss_s.item(), "coords": loss_c.item(), "lattice": loss_l.item()}
    return total, parts
