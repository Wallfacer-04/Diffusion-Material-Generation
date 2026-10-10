"""晶体结构读写与张量转换（v2）。

三部分的表示，以及各自的扩散方式：

  - 原子种类：**离散类别**（整数 0..K-1），走 D3PM 离散扩散
  - 坐标：分数坐标 (N, 3)，走连续高斯扩散
  - 晶格：3×3 矩阵（每行一个晶格向量，单位埃，除以 LATTICE_SCALE 压到 O(1)），走连续高斯扩散

与 v1 的区别只在原子种类：v1 把 one-hot 当连续量加高斯噪声、采样时取 argmax；
v2 改成在类别上做真正的离散扩散（D3PM）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ELEMENT_VOCAB = ("O", "Cu", "Ba", "Pr")
ELEMENT_INDEX = {symbol: i for i, symbol in enumerate(ELEMENT_VOCAB)}

LATTICE_SCALE = 10.0


@dataclass
class Crystal:
    lattice: np.ndarray       # (3, 3)，每一行是一个晶格向量，单位埃
    species: list[str]        # 长度 N
    frac_coords: np.ndarray   # (N, 3)

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
    """Crystal → (物种 one-hot (N,K), 物种类别编号 (N,), 坐标 (N,3), 缩放的晶格 (3,3))。"""
    n = crystal.num_atoms
    index = np.array([ELEMENT_INDEX[s] for s in crystal.species], dtype=np.int64)
    onehot = np.zeros((n, len(ELEMENT_VOCAB)), dtype=np.float32)
    onehot[np.arange(n), index] = 1.0
    coords = crystal.frac_coords.astype(np.float32)
    lattice = (crystal.lattice / LATTICE_SCALE).astype(np.float32)
    return (
        torch.from_numpy(onehot),
        torch.from_numpy(index),
        torch.from_numpy(coords),
        torch.from_numpy(lattice),
    )


def decode(species_index: torch.Tensor, frac_coords: torch.Tensor, lattice_scaled: torch.Tensor) -> Crystal:
    """(物种类别编号, 坐标, 缩放晶格) → Crystal。坐标 wrap 到 [0,1)，晶格还原量纲。"""
    species = [ELEMENT_VOCAB[int(i)] for i in species_index.detach().cpu().tolist()]
    coords = frac_coords.detach().cpu().numpy() % 1.0
    lattice = lattice_scaled.detach().cpu().numpy() * LATTICE_SCALE
    return Crystal(lattice=lattice.astype(np.float64), species=species, frac_coords=coords.astype(np.float64))


def one_hot(index: torch.Tensor, num_elements: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(index, num_elements).to(torch.float32)
