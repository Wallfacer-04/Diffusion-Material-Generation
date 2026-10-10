"""数值验证 v2 的对称性（这是 v2 的核心卖点，必须能自己跑出来）：

    python 14_晶体扩散等变GNN/tests/test_symmetry.py

包含：
  A. 旋转不变：晶格的 Gram 表示对旋转 L -> L·R 不变
  B. 置换等变：关掉槽位嵌入时，打乱原子顺序，输出同步被打乱
  B2. 槽位嵌入（默认）：置换等变被有意打破（单结构重建需要它）
  C. 周期性：整数平移不改变边特征（最小镜像正确）
  D. 解码往返：把真实张量解码回去能还原结构和晶格
  E. 基本流程：前向 / 一步训练 / 采样形状都正常
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gram import gram6_from_lattice
from src.data.graph import periodic_edges
from src.data.structure import ELEMENT_VOCAB, decode, load_crystal, to_tensors
from src.diffusion.forward import training_loss
from src.diffusion.reverse import sample
from src.diffusion.schedule import CosineSchedule
from src.models.egnn import CrystalEGNN


def random_rotation() -> np.ndarray:
    q, r = np.linalg.qr(np.random.randn(3, 3))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    crystal = load_crystal(PROJECT_ROOT / "data" / "Ba2PrCu3O7.json")
    species, coords, gram = to_tensors(crystal)
    n = crystal.num_atoms
    k = len(ELEMENT_VOCAB)
    t0 = torch.zeros(1, dtype=torch.long)

    # ---- A. 旋转不变（Gram 表示）----
    target = gram6_from_lattice(crystal.lattice)
    for _ in range(5):
        rotated = gram6_from_lattice(crystal.lattice @ random_rotation())
        assert np.allclose(rotated, target, atol=1e-8), "Gram 在旋转下应保持不变"
    print("A. 旋转不变 OK：晶格任意旋转，Gram 表示的差异 < 1e-8")

    # ---- B. 置换等变（关掉槽位嵌入）----
    plain = CrystalEGNN(n, k, hidden=32, num_layers=2, use_slot=False).eval()
    perm = torch.randperm(n)
    with torch.no_grad():
        base = plain(species.unsqueeze(0), coords.unsqueeze(0), gram.unsqueeze(0), t0)
        shuffled = plain(species[perm].unsqueeze(0), coords[perm].unsqueeze(0), gram.unsqueeze(0), t0)
    assert torch.allclose(shuffled[0][0], base[0][0][perm], atol=1e-6), "物种输出应跟着置换"
    assert torch.allclose(shuffled[1][0], base[1][0][perm], atol=1e-6), "坐标输出应跟着置换"
    assert torch.allclose(shuffled[2][0], base[2][0], atol=1e-6), "Gram 输出是全局量，应不变"
    print("B. 置换等变 OK：use_slot=False 时，原子重排 → 逐原子输出同步重排")

    # ---- B2. 默认槽位嵌入会打破置换对称（这是刻意的）----
    slotted = CrystalEGNN(n, k, hidden=32, num_layers=2, use_slot=True).eval()
    with torch.no_grad():
        slotted_base = slotted(species.unsqueeze(0), coords.unsqueeze(0), gram.unsqueeze(0), t0)
        slotted_shuffled = slotted(species[perm].unsqueeze(0), coords[perm].unsqueeze(0), gram.unsqueeze(0), t0)
    assert not torch.allclose(slotted_shuffled[1][0], slotted_base[1][0][perm], atol=1e-4)
    print("B2. 槽位嵌入（默认）OK：置换等变被有意打破（单结构重建依赖格位身份）")

    # ---- C. 周期性（最小镜像）----
    edge_a, _, _ = periodic_edges(coords.unsqueeze(0), gram.unsqueeze(0), plain.rbf_centers, plain.rbf_width)
    shift = torch.randint(0, 5, (1, n, 3)).float()
    edge_b, _, _ = periodic_edges((coords.unsqueeze(0) + shift), gram.unsqueeze(0), plain.rbf_centers, plain.rbf_width)
    assert torch.allclose(edge_a, edge_b, atol=1e-6), "整数平移不应改变边特征"
    print("C. 周期性 OK：给坐标加整数平移，边特征不变")

    # ---- D. 解码往返 ----
    restored = decode(species, coords, gram)
    assert restored.species == crystal.species
    assert np.allclose(restored.frac_coords, crystal.frac_coords, atol=1e-6)
    assert np.allclose(restored.lattice, crystal.lattice, atol=1e-4), "正交晶格应能精确还原"
    print("D. 解码往返 OK：物种/坐标/晶格都能还原")

    # ---- E. 基本流程（用默认模型）----
    schedule = CosineSchedule(50)
    model = CrystalEGNN(n, k, hidden=32, num_layers=2)
    b = 4
    s0 = species.unsqueeze(0).expand(b, -1, -1)
    c0 = coords.unsqueeze(0).expand(b, -1, -1)
    g0 = gram.unsqueeze(0).expand(b, -1)
    ps, pc, pg = model(s0, c0, g0, torch.zeros(b, dtype=torch.long))
    assert ps.shape == (b, n, k) and pc.shape == (b, n, 3) and pg.shape == (b, 6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, _ = training_loss(model, schedule, s0, c0, g0)
    loss.backward()
    optimizer.step()
    out = sample(model, schedule, 2, n, k, "cpu", torch.Generator().manual_seed(1))
    assert out[0].shape == (2, n, k) and out[1].shape == (2, n, 3) and out[2].shape == (2, 6)
    print("E. 基本流程 OK：前向 / 一步训练 / 采样形状都正确")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
