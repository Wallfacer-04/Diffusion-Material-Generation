"""前向加噪与训练目标（epsilon-prediction）。

三部分（物种、坐标、Gram 分量）用同一套余弦调度，各自独立加噪。
物种仍是 one-hot 连续松弛（采样时取 argmax）。
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
    gram0: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """联合噪声预测损失，返回 (总损失, 各部分损失明细)。"""
    batch = species0.shape[0]
    device = species0.device
    t = torch.randint(schedule.num_steps, (batch,), device=device)

    noise_s = torch.randn_like(species0)
    noise_c = torch.randn_like(coords0)
    noise_g = torch.randn_like(gram0)

    species_t = q_sample(species0, t, schedule, noise_s)
    coords_t = q_sample(coords0, t, schedule, noise_c)
    gram_t = q_sample(gram0, t, schedule, noise_g)

    pred_s, pred_c, pred_g = model(species_t, coords_t, gram_t, t)

    ws, wc, wg = weights
    loss_s = F.mse_loss(pred_s, noise_s)
    loss_c = F.mse_loss(pred_c, noise_c)
    loss_g = F.mse_loss(pred_g, noise_g)
    total = ws * loss_s + wc * loss_c + wg * loss_g
    parts = {"species": loss_s.item(), "coords": loss_c.item(), "gram": loss_g.item()}
    return total, parts
