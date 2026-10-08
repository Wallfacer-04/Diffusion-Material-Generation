"""晶体结构的读写与张量转换。

表示方式采用主流做法（MatterGen / DiffCSP 那一支的简化版）：

  - 原子种类：固定元素表上的 one-hot 类别向量，形状 (N, K)
  - 坐标：分数坐标（周期性），形状 (N, 3)
  - 晶格：3x3 矩阵，每一行是一个晶格向量，单位埃

晶格整体除以 LATTICE_SCALE，让三部分数值都落在 O(1) 量级，
这样扩散过程对三部分可以用同一套噪声调度。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# 元素词表：固定顺序（按原子序数从小到大），保证 one-hot 位置稳定。
ELEMENT_VOCAB = ("O", "Cu", "Ba", "Pr")
ELEMENT_INDEX = {symbol: i for i, symbol in enumerate(ELEMENT_VOCAB)}
ATOMIC_NUMBER = {"O": 8, "Cu": 29, "Ba": 56, "Pr": 59}

LATTICE_SCALE = 10.0  # 埃；把晶格数值压到 O(1)


@dataclass
class Crystal:
    lattice: np.ndarray       # (3, 3)，每一行是晶格向量，单位埃
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
    """Crystal -> (one-hot (N,K), 分数坐标 (N,3), 缩放后的晶格 (3,3))，均为 float32。"""
    n = crystal.num_atoms
    onehot = np.zeros((n, len(ELEMENT_VOCAB)), dtype=np.float32)
    for i, symbol in enumerate(crystal.species):
        onehot[i, ELEMENT_INDEX[symbol]] = 1.0
    coords = crystal.frac_coords.astype(np.float32)
    lattice = (crystal.lattice / LATTICE_SCALE).astype(np.float32)
    return (
        torch.from_numpy(onehot),
        torch.from_numpy(coords),
        torch.from_numpy(lattice),
    )


def decode(species_scores: torch.Tensor, frac_coords: torch.Tensor, lattice_scaled: torch.Tensor) -> Crystal:
    """连续张量 -> Crystal。原子种类取 argmax，坐标 wrap 到 [0,1)，晶格还原量纲。"""
    species_idx = species_scores.detach().cpu().argmax(dim=-1).tolist()
    species = [ELEMENT_VOCAB[i] for i in species_idx]
    coords = frac_coords.detach().cpu().numpy() % 1.0
    lattice = lattice_scaled.detach().cpu().numpy() * LATTICE_SCALE
    return Crystal(
        lattice=lattice.astype(np.float64),
        species=species,
        frac_coords=coords.astype(np.float64),
    )
