"""把生成的结构写成 VASP POSCAR 文件（纯文本，不依赖 pymatgen/ase）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_poscar(
    path: str | Path,
    lattice: np.ndarray,
    species: list[str],
    frac_coords: np.ndarray,
    comment: str = "Generated structure",
) -> None:
    order: list[str] = []
    for symbol in species:
        if symbol not in order:
            order.append(symbol)
    counts = [species.count(symbol) for symbol in order]

    lines = [comment, "1.0"]
    for row in np.asarray(lattice, dtype=float):
        lines.append("  {:>18.10f} {:>18.10f} {:>18.10f}".format(*row))
    lines.append("  " + "  ".join(order))
    lines.append("  " + "  ".join(str(c) for c in counts))
    lines.append("Direct")
    coords = np.asarray(frac_coords, dtype=float)
    for symbol in order:
        for i, s in enumerate(species):
            if s == symbol:
                lines.append("  {:>18.10f} {:>18.10f} {:>18.10f}".format(*coords[i]))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
