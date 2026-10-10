"""D3PM 离散扩散（Austin et al. 2021）——用在原子种类上。

支持两种前向转移，用 `mode` 选择：

1. **mask（吸收型，默认）**：以概率 β_t 把类别替换成 [MASK]，替换后保持不变。
   边际：q(x_t | x_0) = alpha_bar_t·δ(x_t, x_0) + (1-alpha_bar_t)·δ(x_t, MASK)。
   反向（采样）时：
     - 若 x_t 不是 [MASK]，则上一步必然等于它：x_{t-1} = x_t（**确定性**）；
     - 若 x_t 是 [MASK]，则按后验在「真实类别」和「[MASK]」之间重抽。
   因为大部分位置每步是确定性的，注入共享主干的随机性最小。

2. **uniform（均匀型）**：以概率 β_t 跳到均匀随机的类别，任何位置每步都可能被改写。
   边际：以 alpha_bar_t 停留在 x_0，否则均匀分布在 K 个类别上。

两种都用同一条余弦曲线作为累积保留概率 alpha_bar_t。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


# ---------------------------------------------------------------
# 前向加噪
# ---------------------------------------------------------------

def _keep_prob(alpha_bar: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    prob = alpha_bar[t].to(like.device)
    while prob.dim() < like.dim():
        prob = prob.unsqueeze(-1)
    return prob


def q_sample_uniform(x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor, num_elements: int, generator=None) -> torch.Tensor:
    """均匀型：以概率 alpha_bar_t 保留 x_0，否则均匀随机跳到某个类别。"""
    keep = torch.rand(x0.shape, device=x0.device, generator=generator) < _keep_prob(alpha_bar, t, x0)
    uniform = torch.randint(0, num_elements, x0.shape, device=x0.device, generator=generator)
    return torch.where(keep, x0, uniform)


def q_sample_mask(x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor, num_elements: int, generator=None) -> torch.Tensor:
    """吸收型：以概率 alpha_bar_t 保留 x_0，否则替换成 [MASK]（编号 = num_elements）。"""
    keep = torch.rand(x0.shape, device=x0.device, generator=generator) < _keep_prob(alpha_bar, t, x0)
    mask = torch.full_like(x0, num_elements)
    return torch.where(keep, x0, mask)


def q_sample(x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor, num_elements: int, mode: str = "mask", generator=None) -> torch.Tensor:
    if mode == "mask":
        return q_sample_mask(x0, t, alpha_bar, num_elements, generator)
    if mode == "uniform":
        return q_sample_uniform(x0, t, alpha_bar, num_elements, generator)
    raise ValueError(f"未知的物种扩散类型: {mode}")


def num_species_input(num_elements: int, mode: str) -> int:
    """网络输入的 one-hot 宽度：mask 型要多一维表示 [MASK]。"""
    return num_elements + (1 if mode == "mask" else 0)


def species_one_hot(xt: torch.Tensor, num_elements: int, mode: str) -> torch.Tensor:
    return F.one_hot(xt, num_species_input(num_elements, mode)).to(torch.float32)


# ---------------------------------------------------------------
# 反向采样
# ---------------------------------------------------------------

def posterior_uniform_probs(
    x0_probs: torch.Tensor,
    xt: torch.Tensor,
    alpha_bar_prev: torch.Tensor,
    beta_t: torch.Tensor,
    num_elements: int,
) -> torch.Tensor:
    """均匀型的 q(x_{t-1} | x_t, x_0)，再对预测的 x_0 分布求和，返回 (..., K)。"""
    jump = beta_t / num_elements
    coef = torch.full_like(x0_probs, float(jump))
    index = xt.unsqueeze(-1)
    same_class = torch.as_tensor((1 - beta_t) + jump, dtype=coef.dtype, device=coef.device)
    coef = coef.scatter(-1, index, same_class.expand_as(index))
    base = alpha_bar_prev * x0_probs + (1 - alpha_bar_prev) / num_elements
    unnormalized = coef * base
    return unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def mask_reverse_step(
    x0_probs: torch.Tensor,
    xt: torch.Tensor,
    alpha_bar_prev: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    num_elements: int,
    generator: torch.Generator | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """吸收型的一步反向：非 [MASK] 位置保持不变，[MASK] 位置按后验重抽。

    后验（x_t = MASK 时）：
        P(x_{t-1} = 真实类别 k) = (alpha_bar_{t-1} - alpha_bar_t) / (1 - alpha_bar_t) · p_theta(k)
        P(x_{t-1} = MASK)      = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    """
    mask_index = num_elements
    weight = (alpha_bar_prev - alpha_bar_t) / (1 - alpha_bar_t).clamp_min(1e-12)
    real = weight * x0_probs
    stay_mask = (1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp_min(1e-12)
    probs = torch.cat([real, stay_mask * torch.ones_like(real[..., :1])], dim=-1)   # (..., K+1)
    flat = probs.reshape(-1, num_elements + 1)
    if deterministic:
        sampled = flat.argmax(dim=-1).reshape(xt.shape)
    else:
        sampled = torch.multinomial(flat, 1, generator=generator).reshape(xt.shape)
    return torch.where(xt == mask_index, sampled, xt)


# ---------------------------------------------------------------
# 损失
# ---------------------------------------------------------------

def d3pm_loss(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """网络预测干净类别 x_0（K 个 logits），用交叉熵监督。"""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x0.reshape(-1))
