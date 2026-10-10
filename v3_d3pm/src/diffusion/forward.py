"""训练目标：连续部分（坐标、晶格）做 ε-prediction，离散部分（物种）做 D3PM 交叉熵。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .d3pm import d3pm_loss, q_sample as q_sample_species, species_one_hot
from .schedule import CosineSchedule


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: CosineSchedule, noise: torch.Tensor | None = None) -> torch.Tensor:
    """连续部分的前向加噪：x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*ε。"""
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
    mode: str = "mask",
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """返回 (总损失, 各部分明细)。mode 选择物种扩散类型（mask / uniform）。"""
    batch = coords0.shape[0]
    num_elements = model.num_elements
    device = coords0.device
    t = torch.randint(schedule.num_steps, (batch,), device=device)

    # 连续部分
    noise_c = torch.randn_like(coords0)
    noise_l = torch.randn_like(lattice0)
    coords_t = q_sample(coords0, t, schedule, noise_c)
    lattice_t = q_sample(lattice0, t, schedule, noise_l)

    # 离散部分（D3PM）
    species_t = q_sample_species(species0, t, schedule.alpha_bar, num_elements, mode)
    species_onehot = species_one_hot(species_t, num_elements, mode).to(coords0.dtype)

    logits, pred_c, pred_l = model(species_onehot, coords_t, lattice_t, t)

    ws, wc, wl = weights
    loss_s = d3pm_loss(logits, species0)
    loss_c = F.mse_loss(pred_c, noise_c)
    loss_l = F.mse_loss(pred_l, noise_l)
    total = ws * loss_s + wc * loss_c + wl * loss_l
    parts = {"species_ce": loss_s.item(), "coords_mse": loss_c.item(), "lattice_mse": loss_l.item()}
    return total, parts
