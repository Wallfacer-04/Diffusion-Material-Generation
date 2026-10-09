"""余弦噪声调度（Nichol & Dhariwal），训练和采样共用同一套系数。
把线性 beta 换成余弦调度，噪声注入更平滑。
时间步从 0 到 T-1，其中 0 是最早的加噪层，其反向步不加随机噪声。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class CosineSchedule(nn.Module):
    def __init__(self, num_steps: int = 200):
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps 必须 >= 2")
        self.num_steps = num_steps

        steps = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64)
        curve = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        curve = curve / curve[0]
        betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))

        buffers = {
            "alpha_bar": alpha_bar,
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1 - alpha_bar).sqrt(),
            "posterior_variance": betas * (1 - alpha_bar_prev) / (1 - alpha_bar),
            "posterior_mean_coef1": betas * alpha_bar_prev.sqrt() / (1 - alpha_bar),
            "posterior_mean_coef2": (1 - alpha_bar_prev) * alphas.sqrt() / (1 - alpha_bar),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value.float())

    @staticmethod
    def gather(values: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """按 batch 取系数，并广播到后面的维度（对 (B,·,·) 都适用）。"""
        return values[t].to(dtype=like.dtype, device=like.device).view(-1, 1, 1)
