"""反向采样：物种走 D3PM（mask 或 uniform），坐标与晶格走 DDPM 解析后验。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .d3pm import mask_reverse_step, posterior_uniform_probs
from .schedule import CosineSchedule


@torch.no_grad()
def _reverse_step_continuous(
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
    species_argmax: bool = False,
    mode: str = "mask",
):
    """从纯噪声出发，生成 (物种编号 (B,N), 坐标 (B,N,3), 晶格 (B,3,3))。

    mode: "mask"（吸收型，默认）或 "uniform"（均匀型）。
    species_argmax=True 时物种后验取 argmax（确定性），实测坐标更准、晶格略差。
    """
    s = schedule
    is_mask = mode == "mask"
    if is_mask:
        # 吸收型从「全 MASK」起步
        species = torch.full((num_samples, num_atoms), num_elements, dtype=torch.long, device=device)
    else:
        species = torch.randint(0, num_elements, (num_samples, num_atoms), device=device, generator=generator)
    coords = torch.randn(num_samples, num_atoms, 3, device=device, generator=generator)
    lattice = torch.randn(num_samples, 3, 3, device=device, generator=generator)

    was_training = model.training
    model.eval()
    try:
        for step in reversed(range(s.num_steps)):
            t = torch.full((num_samples,), step, dtype=torch.long, device=device)
            logits, pred_c, pred_l = model(species, coords, lattice, t)

            coords = _reverse_step_continuous(coords, pred_c, t, s, generator)
            lattice = _reverse_step_continuous(lattice, pred_l, t, s, generator)

            p_x0 = F.softmax(logits, dim=-1)
            if step == 0:
                species = p_x0.argmax(dim=-1)
            elif is_mask:
                species = mask_reverse_step(
                    p_x0,
                    species,
                    s.alpha_bar[step - 1],
                    s.alpha_bar[step],
                    num_elements,
                    generator=generator,
                    deterministic=species_argmax,
                )
            elif species_argmax:
                species = posterior_uniform_probs(
                    p_x0, species, s.alpha_bar[step - 1], s.betas[step], num_elements
                ).argmax(dim=-1)
            else:
                posterior = posterior_uniform_probs(
                    p_x0, species, s.alpha_bar[step - 1], s.betas[step], num_elements
                )
                species = torch.multinomial(
                    posterior.reshape(-1, num_elements), 1, generator=generator
                ).reshape(num_samples, num_atoms)
    finally:
        model.train(was_training)
    return species, coords, lattice
