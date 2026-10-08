"""Print and verify the direct embedding of the structure in data/.

Run from any working directory:
    python path/to/single-stru/scripts/test/utils/embed_direct_test.py
Optionally supply another structure file as the first argument.
"""

import argparse
from pathlib import Path
import sys

import torch

# Locate single-stru independently of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.embedding_direct import DirectEmbed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structure",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "Ba2PrCu3O7.json",
        help="Input structure file (default: data/Ba2PrCu3O7.json).",
    )
    args = parser.parse_args()
    embedder = DirectEmbed()
    A, X, L = embedder.read_structure(args.structure)
    tokens = embedder.encode(A, X, L)
    n = A.shape[0]

    # Check the representation and that decoding recovers the input exactly.
    assert tokens.shape == (2 * n + 3, 3), "Unexpected embedding shape"
    assert torch.isfinite(tokens).all(), "Embedding contains NaN or infinity"
    expected_digits = torch.stack((A // 25, (A // 5) % 5, A % 5), dim=-1)
    assert torch.equal(tokens[:n], expected_digits.to(tokens.dtype))
    assert torch.equal(tokens[n:2*n], X), "Fractional coordinates changed"
    assert torch.equal(tokens[2*n:], L), "Lattice vectors changed"
    decoded = embedder.decode(tokens)
    for actual, expected in zip(decoded, (A, X, L)):
        assert torch.equal(actual, expected), "Round-trip decoding failed"

    torch.set_printoptions(precision=6, sci_mode=False, threshold=tokens.numel())
    print(f"Structure: {args.structure.resolve()}")
    print(f"Atoms: {n}")
    print(f"Embedding shape: {tuple(tokens.shape)}")
    print(f"Rows: A [0:{n}], X [{n}:{2*n}], L [{2*n}:{2*n+3}]")
    print("Embedding [A, X, L]:")
    print(tokens)
    print("PASS: shape, finite values, base-5 codes, unchanged X/L, exact decoding")


if __name__ == "__main__":
    main()
