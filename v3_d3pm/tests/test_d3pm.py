"""验证 D3PM 的数学与基本流程（mask 与 uniform 两种都测）：

    python 15_晶体扩散D3PM/tests/test_d3pm.py

包含：
  A. 前向边缘分布：mask 型保留概率应为 alpha_bar_t；uniform 型为 alpha_bar_t + (1-alpha_bar_t)/K
  B. 后验合法性：概率非负、按类别求和为 1
  C. mask 型的确定性：x_t 不是 [MASK] 时，反向一步等于保持原值
  D. oracle 全链路：
       mask：被解开的每个位置都必须是正确类别
       uniform：恢复正确率 > 0.9
  E. 基本流程：前向 / 一步训练 / 采样（两种模式）形状与取值都正确
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.structure import ELEMENT_VOCAB, load_crystal, to_tensors
from src.diffusion.d3pm import (
    mask_reverse_step,
    num_species_input,
    posterior_uniform_probs,
    q_sample_uniform,
    q_sample_mask,
)
from src.diffusion.forward import training_loss
from src.diffusion.reverse import sample
from src.diffusion.schedule import CosineSchedule
from src.models.mpnn import CrystalDenoiser


def main() -> None:
    torch.manual_seed(0)
    K = len(ELEMENT_VOCAB)
    schedule = CosineSchedule(200)
    MASK = K

    # ---- A. 前向边缘分布 ----
    x0 = torch.zeros(20000, 1, dtype=torch.long)
    for t_value in (0, 50, 100, 199):
        t = torch.full((20000,), t_value, dtype=torch.long)
        bar = schedule.alpha_bar[t_value].item()
        y_mask = q_sample_mask(x0, t, schedule.alpha_bar, K)
        assert abs((y_mask == 0).float().mean().item() - bar) < 0.02        # mask: 保留概率 = alpha_bar
        y_uni = q_sample_uniform(x0, t, schedule.alpha_bar, K)
        expected = bar + (1 - bar) / K
        assert abs((y_uni == 0).float().mean().item() - expected) < 0.02    # uniform: 加上 1/K 项
    print("A. 前向边缘分布 OK：mask 型 = alpha_bar_t，uniform 型 = alpha_bar_t + (1-alpha_bar_t)/K")

    # ---- B. 后验合法性 ----
    x0_probs = torch.softmax(torch.randn(5, 7, K), dim=-1)
    xt = torch.randint(0, K, (5, 7))
    pu = posterior_uniform_probs(x0_probs, xt, schedule.alpha_bar[100], schedule.betas[120], K)
    assert torch.all(pu >= 0) and torch.allclose(pu.sum(-1), torch.ones(5, 7), atol=1e-5)
    xt_mask = torch.full((5, 7), MASK)
    pm = torch.cat([
        ((schedule.alpha_bar[99] - schedule.alpha_bar[100]) / (1 - schedule.alpha_bar[100])) * x0_probs,
        ((1 - schedule.alpha_bar[99]) / (1 - schedule.alpha_bar[100])) * torch.ones(5, 7, 1),
    ], dim=-1)
    assert torch.allclose(pm.sum(-1), torch.ones(5, 7), atol=1e-5)
    print("B. 后验合法性 OK：两种模式的概率都非负且逐类求和为 1")

    # ---- C. mask 型反向的确定性 ----
    keep_state = torch.randint(0, K, (16,))
    out = mask_reverse_step(x0_probs[:1, :1].expand(16, 1, K), keep_state,
                            schedule.alpha_bar[99], schedule.alpha_bar[100], K)
    assert torch.equal(out, keep_state), "非 [MASK] 位置必须保持原值"
    print("C. mask 型确定性 OK：x_t 不是 [MASK] 的位置，反向一步保持不变")

    # ---- D. oracle 全链路 ----
    x0_true = torch.randint(0, K, (64,))
    gen = torch.Generator().manual_seed(4)
    species = torch.full(x0_true.shape, MASK, dtype=torch.long)
    for step in reversed(range(1, schedule.num_steps)):
        species = mask_reverse_step(
            F.one_hot(x0_true, K).float(), species,
            schedule.alpha_bar[step - 1], schedule.alpha_bar[step], K, generator=gen,
        )
    unmasked = species != MASK
    assert unmasked.sum().item() > 0
    assert torch.equal(species[unmasked], x0_true[unmasked]), "被解开的位置必须等于真值"
    print(f"D1. mask oracle OK：解开了 {int(unmasked.sum())}/{len(x0_true)} 个位置且全部正确")

    species = torch.randint(0, K, x0_true.shape, generator=torch.Generator().manual_seed(3))
    for step in reversed(range(1, schedule.num_steps)):
        post = posterior_uniform_probs(
            F.one_hot(x0_true, K).float(), species, schedule.alpha_bar[step - 1], schedule.betas[step], K
        )
        species = torch.multinomial(post.reshape(-1, K), 1, generator=gen).reshape(x0_true.shape)
    accuracy = (species == x0_true).float().mean().item()
    assert accuracy > 0.9, accuracy
    print(f"D2. uniform oracle OK：反向采样恢复正确率 {accuracy:.3f}")

    # ---- E. 基本流程（两种模式）----
    crystal = load_crystal(PROJECT_ROOT / "data" / "Ba2PrCu3O7.json")
    _, index0, coords0, lattice0 = to_tensors(crystal)
    n = crystal.num_atoms
    b = 4
    bi = index0.unsqueeze(0).expand(b, -1)
    bc = coords0.unsqueeze(0).expand(b, -1, -1)
    bl = lattice0.unsqueeze(0).expand(b, -1, -1)
    for mode in ("mask", "uniform"):
        model = CrystalDenoiser(
            n, K, num_species_input=num_species_input(K, mode), hidden=32, num_layers=2
        )
        loss, _ = training_loss(model, schedule, bi, bc, bl, mode)
        loss.backward()
        assert torch.isfinite(loss)
        out = sample(model, schedule, 2, n, K, "cpu", torch.Generator().manual_seed(1), mode=mode)
        assert out[0].shape == (2, n) and out[0].dtype == torch.long
        assert int(out[0].min()) >= 0 and int(out[0].max()) < K
        assert out[1].shape == (2, n, 3) and out[2].shape == (2, 3, 3)
    print("E. 基本流程 OK：mask / uniform 两种模式的前向、训练、采样都正常")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
