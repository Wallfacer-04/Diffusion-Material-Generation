"""快速自检（不训练完整模型）：
    python 13_晶体扩散GNN周期图/tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.structure import ELEMENT_VOCAB, decode, load_crystal, to_tensors
from src.diffusion.forward import q_sample, training_loss
from src.diffusion.reverse import sample
from src.diffusion.schedule import CosineSchedule
from src.models.mpnn import CrystalDenoiser
from src.utils.poscar import write_poscar


def main() -> None:
    torch.manual_seed(0)
    crystal = load_crystal(PROJECT_ROOT / "data" / "Ba2PrCu3O7.json")
    species, coords, lattice = to_tensors(crystal)
    n = crystal.num_atoms
    k = len(ELEMENT_VOCAB)

    assert species.shape == (n, k) and coords.shape == (n, 3) and lattice.shape == (3, 3)
    print("张量形状 OK:", tuple(species.shape), tuple(coords.shape), tuple(lattice.shape))

    # 调度单调递减、末端接近 0
    schedule = CosineSchedule(50)
    assert torch.all(schedule.alpha_bar[1:] < schedule.alpha_bar[:-1])
    assert schedule.alpha_bar[-1] < 1e-2
    assert schedule.posterior_variance[0].item() == 0.0
    print("余弦调度 OK: alpha_bar[-1] =", round(schedule.alpha_bar[-1].item(), 6))

    # 前向公式：x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise
    b = 4
    s0 = species.unsqueeze(0).expand(b, -1, -1)
    c0 = coords.unsqueeze(0).expand(b, -1, -1)
    l0 = lattice.unsqueeze(0).expand(b, -1, -1)
    t0 = torch.zeros(b, dtype=torch.long)
    fixed_noise = torch.randn_like(s0)
    xt = q_sample(s0, t0, schedule, fixed_noise)
    expected = schedule.sqrt_alpha_bar[0] * s0 + schedule.sqrt_one_minus_alpha_bar[0] * fixed_noise
    assert torch.allclose(xt, expected, atol=1e-5)
    print("前向公式 OK: x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise")

    # 模型前向 + 训练一步
    model = CrystalDenoiser(n, k, hidden=32, num_layers=2)
    ps, pc, pl = model(s0, c0, l0, torch.zeros(b, dtype=torch.long))
    assert ps.shape == (b, n, k) and pc.shape == (b, n, 3) and pl.shape == (b, 3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, parts = training_loss(model, schedule, s0, c0, l0)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    print("模型前向+训练一步 OK, loss =", round(loss.item(), 5))

    # 采样 + 解码 + 写 POSCAR
    gen = sample(model, schedule, 2, n, k, "cpu", torch.Generator().manual_seed(1))
    generated = decode(*[g[0] for g in gen])
    write_poscar(PROJECT_ROOT / "outputs" / "_smoke.vasp", generated.lattice,
                 generated.species, generated.frac_coords, comment="smoke test")
    assert len(generated.species) == n
    print("采样+解码+POSCAR OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
