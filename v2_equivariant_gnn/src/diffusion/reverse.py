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
    x0 = (xt - s.gather(s.sqrt_one_minus_alpha_bar, t, xt) * pred) / s.gather(s.sqrt_alpha_bar, t, xt)
    mean = s.gather(s.posterior_mean_coef1, t, xt) * x0 + s.gather(s.posterior_mean_coef2, t, xt) * xt
    var = s.gather(s.posterior_variance, t, xt)
    noise = torch.randn(xt.shape, device=xt.device, dtype=xt.dtype, generator=generator)
    factor = (t > 0).view(-1, *([1] * (xt.dim() - 1)))
    return mean + factor * var.sqrt() * noise


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
    """从纯高斯噪声出发，生成 (物种, 坐标, Gram6) 三部分连续张量。"""
    s = schedule
    species = torch.randn(num_samples, num_atoms, num_elements, device=device, generator=generator)
    coords = torch.randn(num_samples, num_atoms, 3, device=device, generator=generator)
    gram = torch.randn(num_samples, 6, device=device, generator=generator)

    was_training = model.training
    model.eval()
    try:
        for step in reversed(range(s.num_steps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=device)
            pred_s, pred_c, pred_g = model(species, coords, gram, t)
            species = _reverse_step(species, pred_s, t, s, generator)
            coords = _reverse_step(coords, pred_c, t, s, generator)
            gram = _reverse_step(gram, pred_g, t, s, generator)
    finally:
        model.train(was_training)
    return species, coords, gram
