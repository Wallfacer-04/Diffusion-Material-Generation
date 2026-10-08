"""Training-time forward noising and the noise-prediction objective."""

import torch
from torch import nn
from torch.nn import functional as F

from .schedule import DiffusionSchedule


class ForwardDiffusion(nn.Module):
    def __init__(self, schedule: DiffusionSchedule):
        super().__init__()
        self.schedule = schedule

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Draw x_t directly from q(x_t | x_0), without a sequential noise loop."""
        self.schedule.validate(x0, t)
        if noise is None:
            noise = torch.randn_like(x0)
        if noise.shape != x0.shape or noise.device != x0.device or noise.dtype != x0.dtype:
            raise ValueError("noise must match x0's shape, device, and dtype")
        s = self.schedule
        return s.extract(s.sqrt_alpha_bar, t, x0) * x0 + s.extract(
            s.sqrt_one_minus_alpha_bar, t, x0
        ) * noise

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return mean epsilon MSE; the caller owns backward() and optimizer.step()."""
        if t is None:
            t = torch.randint(self.schedule.num_steps, (x0.shape[0],), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        prediction = model(xt, t)
        if prediction.shape != x0.shape:
            raise ValueError("model must predict noise with the same shape as x0")
        return F.mse_loss(prediction, noise)
