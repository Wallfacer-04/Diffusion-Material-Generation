"""DDPM 反向采样：用解析后验均值与方差逐步去噪。"""

from __future__ import annotations

import torch
from torch import nn

from .schedule import CosineSchedule


@torch.no_grad()
def _reverse_step(
    xt: torch.Tensor,
    pred: torch.Tensor,
    t: torch.Tensor,
    schedule: CosineSchedule,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    s = schedule
    # 先用预测噪声反推 x0
    x0 = (xt - s.gather(s.sqrt_one_minus_alpha_bar, t, xt) * pred) / s.gather(s.sqrt_alpha_bar, t, xt)
    mean = s.gather(s.posterior_mean_coef1, t, xt) * x0 + s.gather(s.posterior_mean_coef2, t, xt) * xt
    var = s.gather(s.posterior_variance, t, xt)
    noise = torch.randn(xt.shape, device=xt.device, dtype=xt.dtype, generator=generator)
    return mean + (t > 0).view(-1, 1, 1) * var.sqrt() * noise


@torch.no_grad()
def sample(
    model: nn.Module,
    schedule: CosineSchedule,
    num_samples: int,
    num_atoms: int,
    num_elements: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
):
    """从纯高斯噪声出发，生成 (物种, 坐标, 晶格) 三部分连续张量。"""
    s = schedule
    species = torch.randn(num_samples, num_atoms, num_elements, device=device, generator=generator)
    coords = torch.randn(num_samples, num_atoms, 3, device=device, generator=generator)
    lattice = torch.randn(num_samples, 3, 3, device=device, generator=generator)

    was_training = model.training
    model.eval()
    try:
        for step in reversed(range(s.num_steps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=device)
            pred_s, pred_c, pred_l = model(species, coords, lattice, t)
            species = _reverse_step(species, pred_s, t, s, generator)
            coords = _reverse_step(coords, pred_c, t, s, generator)
            lattice = _reverse_step(lattice, pred_l, t, s, generator)
    finally:
        model.train(was_training)
    return species, coords, lattice
