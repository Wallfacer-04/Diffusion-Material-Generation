"""在单个晶体结构上训练周期图扩散模型，再从噪声采样生成它自己。

用法（在 diffusion 环境下，仓库根目录随意）：
    python 13_晶体扩散GNN周期图/scripts/train.py --steps 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.structure import ELEMENT_VOCAB, decode, load_crystal, to_tensors
from src.diffusion.forward import training_loss
from src.diffusion.reverse import sample
from src.diffusion.schedule import CosineSchedule
from src.models.mpnn import CrystalDenoiser
from src.utils.poscar import write_poscar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=PROJECT_ROOT / "data" / "Ba2PrCu3O7.json")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=200, help="扩散总步数 T")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = "cpu"

    crystal = load_crystal(args.structure)
    species0, coords0, lattice0 = to_tensors(crystal)
    n = crystal.num_atoms
    k = len(ELEMENT_VOCAB)
    print(f"结构: {crystal.num_atoms} 个原子, {k} 种元素, 物种张量 {tuple(species0.shape)}")

    # 单结构 -> 复制成 batch（每份加不同的噪声）
    batch_s = species0.unsqueeze(0).expand(args.batch_size, -1, -1).contiguous()
    batch_c = coords0.unsqueeze(0).expand(args.batch_size, -1, -1).contiguous()
    batch_l = lattice0.unsqueeze(0).expand(args.batch_size, -1, -1).contiguous()

    model = CrystalDenoiser(n, k, hidden=args.hidden, num_layers=args.layers)
    print("模型参数量:", sum(p.numel() for p in model.parameters()))
    schedule = CosineSchedule(args.num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 训练 ----
    for step in range(args.steps):
        optimizer.zero_grad()
        loss, parts = training_loss(model, schedule, batch_s, batch_c, batch_l)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 200 == 0:
            print(
                f"update {step + 1:5d}  loss={loss.item():.5f}  "
                f"(物种 {parts['species']:.5f}  坐标 {parts['coords']:.5f}  晶格 {parts['lattice']:.5f})"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "steps": args.steps, "num_steps": args.num_steps},
        args.output_dir / "crystal_gnn.pt",
    )

    # ---- 采样生成 ----
    generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    gen_s, gen_c, gen_l = sample(model, schedule, args.samples, n, k, device, generator)

    # ---- 解码 + 导出 POSCAR + 误差报告 ----
    print("\n生成结果（对比真实结构）:")
    for i in range(args.samples):
        generated = decode(gen_s[i], gen_c[i], gen_l[i])
        write_poscar(
            args.output_dir / f"sample_{i:04d}.vasp",
            generated.lattice,
            generated.species,
            generated.frac_coords,
            comment=f"Generated sample {i}",
        )
        coord_err = float(abs(generated.frac_coords - crystal.frac_coords).mean())
        lat_err = float(abs(generated.lattice - crystal.lattice).mean())
        match = sum(a == b for a, b in zip(generated.species, crystal.species))
        print(
            f"sample {i}: 物种匹配 {match}/{n}  "
            f"坐标 MAE {coord_err:.4f}  晶格 MAE {lat_err:.4f} 埃"
        )

    print(f"\n模型和 {args.samples} 个 POSCAR 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
