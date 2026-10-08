"""DDPM reverse denoising with fixed posterior variance and no clipping."""

import torch
from torch import nn

from .schedule import DiffusionSchedule


class ReverseDiffusion(nn.Module):
    def __init__(self, schedule: DiffusionSchedule):
        super().__init__()
        self.schedule = schedule

    def p_mean_variance(
        self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the reverse mean (B, tokens, 3) and variance (B, 1, 1)."""
        s = self.schedule
        s.validate(xt, t)
        epsilon = model(xt, t)
        if epsilon.shape != xt.shape:
            raise ValueError("model must predict noise with the same shape as xt")
        x0 = (xt - s.extract(s.sqrt_one_minus_alpha_bar, t, xt) * epsilon) / s.extract(
            s.sqrt_alpha_bar, t, xt
        )
        mean = s.extract(s.posterior_mean_coef1, t, xt) * x0 + s.extract(
            s.posterior_mean_coef2, t, xt
        ) * xt
        return mean, s.extract(s.posterior_variance, t, xt)

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Take one reverse step. Caller sets model.eval() for standalone use."""
        mean, variance = self.p_mean_variance(model, xt, t)
        if torch.all(t == 0):
            return mean
        noise = torch.randn(xt.shape, device=xt.device, dtype=xt.dtype, generator=generator)
        return mean + (t > 0).view(-1, 1, 1) * variance.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample continuous scaled tokens using every step, from T-1 down to 0.

        The schedule determines device/dtype. A supplied generator must match
        that device. Preserve all prior model training/evaluation flags.
        """
        if len(shape) != 3 or shape[-1] != 3 or any(n < 1 for n in shape):
            raise ValueError("shape must be (B, tokens, 3), with B, tokens > 0")
        states = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            xt = torch.randn(
                shape, device=self.schedule.betas.device,
                dtype=self.schedule.betas.dtype, generator=generator,
            )
            for step in reversed(range(self.schedule.num_steps)):
                t = torch.full((shape[0],), step, device=xt.device, dtype=torch.long)
                xt = self.p_sample(model, xt, t, generator)
            return xt
        finally:
            for module, training in states:
                module.training = training
