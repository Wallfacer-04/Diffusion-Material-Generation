"""A minimal diffusion transformer block using adaLN-Zero conditioning."""

import math

import torch
from torch import nn


class DiTBlock(nn.Module):
    """Conditioned self-attention and MLP with gated residual connections.

    Args:
        hidden_size: Token and conditioning embedding width.
        num_heads: Number of attention heads; must divide hidden_size.
        mlp_ratio: MLP intermediate width relative to hidden_size.

    Inputs are tokens x of shape (batch, tokens, hidden_size) and an already
    embedded condition c of shape (batch, hidden_size), e.g. a timestep embedding
    optionally summed with a class embedding. Input projection, positional
    encoding, and timestep embedding belong outside this block.

    Zero-initialized modulation makes the block an identity at initialization.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0
    ):
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        mlp_size = int(hidden_size * mlp_ratio)
        if mlp_size <= 0:
            raise ValueError("mlp_ratio must produce a positive intermediate width")

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_size, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Return updated tokens with the same shape as x."""
        if x.ndim != 3 or x.shape[-1] != self.attn.embed_dim:
            raise ValueError("x must have shape (batch, tokens, hidden_size)")
        if c.shape != (x.shape[0], x.shape[2]):
            raise ValueError("c must have shape (batch, hidden_size) matching x")
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = self._modulate(self.norm1(x), shift_a, scale_a)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a.unsqueeze(1) * h
        h = self._modulate(self.norm2(x), shift_m, scale_m)
        return x + gate_m.unsqueeze(1) * self.mlp(h)


class DiTDenoiser(nn.Module):
    """Predict three noise values per token for a fixed token count/order.

    Diffusion remains in (B, num_tokens, 3) space. The learned input/output
    projections are internal features, not an invertible encoding pair.
    """

    def __init__(
        self, num_tokens: int, hidden_size: int = 32, num_heads: int = 4,
        depth: int = 2, mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if num_tokens < 1 or depth < 1:
            raise ValueError("num_tokens and depth must be positive")
        if hidden_size < 2 or hidden_size % 2:
            raise ValueError("hidden_size must be even and >= 2 for timestep embeddings")
        self.num_tokens = num_tokens
        self.input_projection = nn.Linear(3, hidden_size)
        self.position_embedding = nn.Parameter(torch.empty(1, num_tokens, hidden_size))
        nn.init.normal_(self.position_embedding, std=0.02)
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(hidden_size // 2).float() / (hidden_size // 2)
        )
        self.register_buffer("time_frequencies", frequencies)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output_projection = nn.Linear(hidden_size, 3)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.num_tokens, 3):
            raise ValueError("x must have shape (B, num_tokens, 3)")
        if t.shape != (x.shape[0],) or t.dtype != torch.long:
            raise ValueError("t must be a torch.long tensor of shape (B,)")
        if t.device != x.device:
            raise ValueError("x and t must share a device")
        angles = t[:, None].to(self.time_frequencies.dtype) * self.time_frequencies[None]
        condition = self.time_mlp(torch.cat((angles.cos(), angles.sin()), dim=-1))
        h = self.input_projection(x) + self.position_embedding
        for block in self.blocks:
            h = block(h, condition)
        return self.output_projection(self.final_norm(h))
