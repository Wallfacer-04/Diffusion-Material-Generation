"""Shared cosine DDPM schedule, with timesteps indexed from 0 to T - 1."""

import math

import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    """Fixed coefficients shared by training and sampling.

    Index 0 is the first noisy level, not clean data. The previous cumulative
    alpha at index 0 is 1, so its reverse posterior variance is exactly zero.
    Move this module and the denoiser to the same device before use.
    """

    def __init__(self, num_steps: int = 100):
        super().__init__()
        if not isinstance(num_steps, int) or num_steps < 2:
            raise ValueError("num_steps must be an integer >= 2")
        self.num_steps = num_steps
        steps = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64)
        cumulative = torch.cos((steps + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        alpha_bar = alphas.cumprod(dim=0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))
        coefficients = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bar": alpha_bar,
            "alpha_bar_prev": previous,
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1 - alpha_bar).sqrt(),
            "posterior_variance": betas * (1 - previous) / (1 - alpha_bar),
            "posterior_mean_coef1": betas * previous.sqrt() / (1 - alpha_bar),
            "posterior_mean_coef2": (1 - previous) * alphas.sqrt() / (1 - alpha_bar),
        }
        for name, value in coefficients.items():
            self.register_buffer(name, value.float())

    def validate(self, x: torch.Tensor, t: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != 3 or min(x.shape[:2]) < 1:
            raise ValueError("tokens must have shape (B, tokens, 3), with B, tokens > 0")
        if not x.is_floating_point():
            raise TypeError("tokens must be floating point")
        if t.shape != (x.shape[0],) or t.dtype != torch.long:
            raise ValueError("timesteps must be a torch.long tensor of shape (B,)")
        if x.device != self.betas.device or t.device != x.device:
            raise ValueError("tokens, timesteps, and schedule must share a device")
        if torch.any((t < 0) | (t >= self.num_steps)):
            raise ValueError("timesteps must be in [0, num_steps)")

    @staticmethod
    def extract(values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gather one coefficient per example for broadcasting over tokens."""
        return values[t].to(dtype=x.dtype).view(-1, 1, 1)
