"""晶体结构读写与张量转换（v2）。

三部分的表示：

  - 原子种类：固定元素表上的 one-hot 类别向量，形状 (N, K)
  - 坐标：分数坐标（周期性），形状 (N, 3)
  - 晶格：用 Gram 矩阵的 6 个分量表示（旋转不变），除以 GRAM_SCALE 压到 O(1)

晶格用 Gram 表示是 v2 的关键改动：它让模型对"晶体朝向"不敏感，
生成出来的晶格与目标只差一个旋转（物理上同一个晶体）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .gram import gram6_from_lattice, lattice_from_gram6

ELEMENT_VOCAB = ("O", "Cu", "Ba", "Pr")
ELEMENT_INDEX = {symbol: i for i, symbol in enumerate(ELEMENT_VOCAB)}

LATTICE_SCALE = 10.0
GRAM_SCALE = LATTICE_SCALE ** 2  # 100.0，把 Gram 分量压到 O(1)


@dataclass
class Crystal:
    lattice: np.ndarray       # (3, 3)，每一行是一个晶格向量，单位埃
    species: list[str]        # 长度 N
    frac_coords: np.ndarray   # (N, 3)，分数坐标

    @property
    def num_atoms(self) -> int:
        return len(self.species)


def load_crystal(path: str | Path) -> Crystal:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Crystal(
        lattice=np.asarray(payload["lattice"], dtype=np.float64),
        species=list(payload["species"]),
        frac_coords=np.asarray(payload["frac_coords"], dtype=np.float64),
    )


def to_tensors(crystal: Crystal):
    """Crystal → (one-hot (N,K), 坐标 (N,3), Gram6 (6,))，均为 float32。"""
    n = crystal.num_atoms
    onehot = np.zeros((n, len(ELEMENT_VOCAB)), dtype=np.float32)
    for i, symbol in enumerate(crystal.species):
        onehot[i, ELEMENT_INDEX[symbol]] = 1.0
    coords = crystal.frac_coords.astype(np.float32)
    gram6 = (gram6_from_lattice(crystal.lattice) / GRAM_SCALE).astype(np.float32)
    return (
        torch.from_numpy(onehot),
        torch.from_numpy(coords),
        torch.from_numpy(gram6),
    )


def decode(species_scores: torch.Tensor, frac_coords: torch.Tensor, gram6_scaled: torch.Tensor) -> Crystal:
    """连续张量 → Crystal。物种取 argmax；坐标 wrap 到 [0,1)；Gram 还原后 Cholesky 得晶格。"""
    species_idx = species_scores.detach().cpu().argmax(dim=-1).tolist()
    species = [ELEMENT_VOCAB[i] for i in species_idx]
    coords = frac_coords.detach().cpu().numpy() % 1.0
    gram6 = gram6_scaled.detach().cpu().numpy() * GRAM_SCALE
    lattice = lattice_from_gram6(gram6)
    return Crystal(lattice=lattice, species=species, frac_coords=coords)
