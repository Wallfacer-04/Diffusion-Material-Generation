"""Fixed [A, X, L] encoding with shape (..., 2*N + 3, 3).

Atomic numbers 0..118 become base-5 digits, most significant first:
Z = 25*a + 5*b + c. Oxygen (8) is (0, 1, 3); 118 is (4, 3, 3).
Zero is a reserved code, not a physical element. Fractional coordinates
and lattice row vectors (angstroms) are copied without normalization.

Usage:
    embedder = DirectEmbed()
    A, X, L = embedder.read_structure("data/Ba2PrCu3O7.json")
    tokens = embedder.encode(A, X, L)
    A, X, L = embedder.decode(tokens)
"""

from pathlib import Path

import torch


class DirectEmbed:

    @staticmethod
    def read_structure(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read an ordered pymatgen structure as CPU A(int64), X/L(float32)."""
        from pymatgen.core import Structure

        structure = Structure.from_file(str(path))
        if not structure.is_ordered or len(structure) == 0:
            raise ValueError("The structure must contain fully occupied, ordered sites")
        A = torch.tensor(structure.atomic_numbers, dtype=torch.long)
        X = torch.tensor(structure.frac_coords, dtype=torch.float32)
        L = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
        return A, X, L

    def __call__(self, A: torch.Tensor, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Allow embedder(A, X, L) as shorthand for encode(A, X, L)."""
        return self.encode(A, X, L)

    def encode(self, A: torch.Tensor, X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Encode (..., N), (..., N, 3), (..., 3, 3) as [N A; N X; 3 L].

        All tensors must share a device; X and L must share a floating dtype.
        Coordinates are not wrapped. Gradients through X and L are preserved.
        """
        if A.ndim < 1 or A.shape[-1] == 0:
            raise ValueError("A must have shape (..., N), with N > 0")
        if X.shape != (*A.shape, 3):
            raise ValueError("X must have shape (..., N, 3) matching A")
        if L.shape != (*A.shape[:-1], 3, 3):
            raise ValueError("L must have shape (..., 3, 3) matching A")
        if A.dtype not in (torch.int32, torch.int64):
            raise TypeError("A must contain integer atomic numbers")
        if torch.any((A < 0) | (A > 118)):
            raise ValueError("Atomic numbers must be between 0 and 118")
        if not X.is_floating_point() or not L.is_floating_point():
            raise TypeError("X and L must be floating-point tensors")
        if X.dtype != L.dtype:
            raise TypeError("X and L must have the same dtype")
        if A.device != X.device or X.device != L.device:
            raise ValueError("A, X and L must share a device")
        if not torch.isfinite(X).all() or not torch.isfinite(L).all():
            raise ValueError("X and L must contain only finite values")
        # Each digit is in 0..4, ordered by place value 25, 5, 1.
        digits = torch.stack((A // 25, (A // 5) % 5, A % 5), dim=-1)
        return torch.cat((digits.to(dtype=X.dtype), X, L), dim=-2)

    @staticmethod
    def decode(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Invert an exact encoding, preserving coordinates and lattice values.

        This rejects noninteger digits and unused codes 119..124. Generated
        continuous samples need an explicit discretization policy before decoding;
        this function deliberately does not round or clamp them silently.
        """
        if tokens.ndim < 2 or tokens.shape[-1] != 3:
            raise ValueError("tokens must have shape (..., 2*N + 3, 3)")
        length = tokens.shape[-2]
        if length < 5 or (length - 3) % 2:
            raise ValueError("Token count must equal 2*N + 3, with N > 0")
        if not tokens.is_floating_point():
            raise TypeError("tokens must be floating-point tensors")
        if not torch.isfinite(tokens).all():
            raise ValueError("tokens must contain only finite values")
        n = (length - 3) // 2
        digits = tokens[..., :n, :]
        if torch.any((digits < 0) | (digits > 4) | (digits != digits.round())):
            raise ValueError("Atom tokens must contain integer base-5 digits in 0..4")
        digits = digits.to(dtype=torch.long)
        A = 25 * digits[..., 0] + 5 * digits[..., 1] + digits[..., 2]
        if torch.any(A > 118):
            raise ValueError("Atom tokens contain unused atomic codes 119..124")
        return A, tokens[..., n:2*n, :], tokens[..., 2*n:, :]
