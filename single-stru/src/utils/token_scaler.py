"""Reversible scaling of DirectEmbed tokens, separate from their encoding."""

import math

import torch
from torch import nn


class TokenScaler(nn.Module):
    """Fit once on training data; reuse the same state for every sample.

    Atom count and lattice scale are buffers included in state_dict(). No atom
    rounding, coordinate wrapping, or prediction clipping is performed.
    """

    def __init__(self, num_atoms: int, lattice_scale: float):
        super().__init__()
        if num_atoms < 1 or not isinstance(num_atoms, int):
            raise ValueError("num_atoms must be a positive integer")
        if not math.isfinite(lattice_scale) or lattice_scale < 1:
            raise ValueError("lattice_scale must be finite and >= 1")
        self.register_buffer("atom_count", torch.tensor(num_atoms, dtype=torch.long))
        self.register_buffer("lattice_scale", torch.tensor(lattice_scale, dtype=torch.float32))

    @property
    def num_atoms(self) -> int:
        return int(self.atom_count.item())

    @classmethod
    def fit(cls, training_tokens: torch.Tensor) -> "TokenScaler":
        """Infer fixed atom count and max(1, max(abs(training lattice)))."""
        if training_tokens.ndim < 2 or training_tokens.shape[-1] != 3:
            raise ValueError("training_tokens must have shape (..., 2*N + 3, 3)")
        length = training_tokens.shape[-2]
        if length < 5 or (length - 3) % 2:
            raise ValueError("token count must equal 2*N + 3, with N > 0")
        if not training_tokens.is_floating_point() or not torch.isfinite(training_tokens).all():
            raise ValueError("training_tokens must be finite floating-point values")
        scale = max(1.0, training_tokens[..., -3:, :].abs().max().item())
        return cls((length - 3) // 2, scale).to(
            device=training_tokens.device, dtype=training_tokens.dtype
        )

    def _validate(self, tokens: torch.Tensor) -> None:
        if tokens.ndim < 2 or tokens.shape[-2:] != (2 * self.num_atoms + 3, 3):
            raise ValueError("tokens must have shape (..., 2*num_atoms + 3, 3)")
        if not tokens.is_floating_point():
            raise TypeError("tokens must be floating point")
        if tokens.device != self.lattice_scale.device:
            raise ValueError("tokens and scaler must share a device")

    def transform(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate(tokens)
        n = self.num_atoms
        return torch.cat((
            tokens[..., :n, :] / 2 - 1,
            tokens[..., n:2*n, :] * 2 - 1,
            tokens[..., 2*n:, :] / self.lattice_scale.to(dtype=tokens.dtype),
        ), dim=-2)

    def inverse_transform(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate(tokens)
        n = self.num_atoms
        return torch.cat((
            (tokens[..., :n, :] + 1) * 2,
            (tokens[..., n:2*n, :] + 1) / 2,
            tokens[..., 2*n:, :] * self.lattice_scale.to(dtype=tokens.dtype),
        ), dim=-2)
