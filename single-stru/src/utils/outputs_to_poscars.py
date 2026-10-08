"""Convert generated DirectEmbed output tensors into VASP POSCAR files.

Instructions (run from the repository root):
    conda activate diffusion
    python single-stru/src/utils/outputs_to_poscars.py single-stru/outputs/diffusion_demo.pt
    python single-stru/src/utils/outputs_to_poscars.py outputs.pt --output-dir poscars

Requires torch and pymatgen. Input is a torch.save checkpoint containing a
``samples`` tensor, as produced by scripts/diffusion_demo.py, or a saved tensor
itself. Supported shapes are (2*N+3, 3) and (B, 2*N+3, 3). Values must already
be in DirectEmbed's original units: base-5 atom digits, fractional coordinates,
and lattice row vectors in angstroms. Do not pass normalized model outputs.

If this Windows environment reports an OpenMP libomp/libiomp5md conflict, set
``$env:MKL_THREADING_LAYER = "SEQUENTIAL"`` in PowerShell before running Python.

Each continuous atom triplet is assigned the nearest valid base-5 encoding
among atomic numbers 1..118 by Euclidean distance (ties choose the lower Z).
This excludes reserved code 0 and unused codes 119..124. Fractional coordinates
are wrapped into [0, 1); lattice vectors are preserved. This projection does
not establish chemical validity or stability, and may change composition.

Outputs default to a ``poscars`` directory beside the input, named
sample_0000.vasp, sample_0001.vasp, etc. All samples are validated before writing;
nonfinite values and singular lattices are rejected. Existing files are refused
unless --overwrite is supplied. --strict-atoms requires exact valid atom codes
instead of projecting them. Files contain Direct (fractional) coordinates.

Python API (with single-stru on sys.path):
    from src.utils.outputs_to_poscars import convert_outputs
    paths = convert_outputs("single-stru/outputs/diffusion_demo.pt")
"""

import argparse
from pathlib import Path

import torch
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar


def samples_to_structures(samples: torch.Tensor, *, strict_atoms: bool = False) -> list[Structure]:
    """Decode original-unit tokens, explicitly projecting atom codes by default."""
    if not isinstance(samples, torch.Tensor) or not samples.is_floating_point():
        raise ValueError("samples must be a floating-point tensor")
    samples = samples.detach().to(device="cpu", dtype=torch.float64)
    if samples.ndim == 2:
        samples = samples.unsqueeze(0)
    if samples.ndim != 3 or samples.shape[-1] != 3 or samples.shape[0] == 0:
        raise ValueError("samples must have shape (2*N+3, 3) or (B, 2*N+3, 3), B > 0")
    length = samples.shape[1]
    if length < 5 or (length - 3) % 2:
        raise ValueError("Token count must equal 2*N+3, N > 0")
    if not torch.isfinite(samples).all():
        raise ValueError("samples contain NaN or infinity")
    n = (length - 3) // 2
    numbers = torch.arange(1, 119)
    codes = torch.stack((numbers // 25, (numbers // 5) % 5, numbers % 5), dim=-1).double()
    structures = []
    for index, sample in enumerate(samples):
        digits = sample[:n]
        if strict_atoms:
            atomic_numbers = (digits * torch.tensor([25, 5, 1])).sum(-1)
            if (torch.any(digits != digits.round()) or torch.any((digits < 0) | (digits > 4))
                    or torch.any((atomic_numbers < 1) | (atomic_numbers > 118))):
                raise ValueError(f"Sample {index}: atom codes must exactly encode elements 1..118")
            atomic_numbers = atomic_numbers.long()
        else:
            atomic_numbers = torch.cdist(digits, codes).argmin(-1) + 1
        lattice = sample[-3:]
        if torch.linalg.matrix_rank(lattice).item() < 3:
            raise ValueError(f"Sample {index}: lattice is singular")
        structures.append(Structure(
            lattice.numpy(), atomic_numbers.tolist(), sample[n:2*n].remainder(1).numpy(),
            coords_are_cartesian=False,
        ))
    return structures


def convert_outputs(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    strict_atoms: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Load a checkpoint on CPU and write one POSCAR per generated sample."""
    input_path = Path(input_path)
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        if "samples" not in payload:
            raise ValueError("Checkpoint does not contain a 'samples' tensor")
        payload = payload["samples"]
    structures = samples_to_structures(payload, strict_atoms=strict_atoms)
    output_dir = Path(output_dir) if output_dir is not None else input_path.parent / "poscars"
    source_dir = Path(__file__).resolve().parents[1]
    if output_dir.resolve().is_relative_to(source_dir):
        raise ValueError("output-dir must be outside src/")
    paths = [output_dir / f"sample_{index:04d}.vasp" for index in range(len(structures))]
    for path in paths:
        if path.resolve() == input_path.resolve():
            raise ValueError("Output must not replace the input checkpoint")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite to replace")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (path, structure) in enumerate(zip(paths, structures)):
        # Keep site order so every species stays paired with its generated position.
        content = Poscar(structure, comment=f"Generated sample {index}", sort_structure=False).get_str(direct=True)
        with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(content)
    return paths


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Checkpoint or original-unit samples tensor (.pt)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strict-atoms", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        paths = convert_outputs(args.input, args.output_dir, strict_atoms=args.strict_atoms, overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {len(paths)} POSCAR files to {paths[0].parent.resolve()}")


if __name__ == "__main__":
    main()
