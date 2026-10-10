"""在单个晶体结构上训练 v2（物种 D3PM 离散扩散 + 坐标/晶格连续扩散），再生成它自己。

用法：
    python 15_晶体扩散D3PM/scripts/train.py --steps 4000 --samples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.structure import ELEMENT_VOCAB, decode, load_crystal, to_tensors
from src.diffusion.d3pm import num_species_input
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
    parser.add_argument("--species-weight", type=float, default=1.0, help="物种交叉熵的权重（相对坐标 MSE）")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--species-argmax", action="store_true", help="物种后验用 argmax（确定性），坐标更准、晶格略差")
    parser.add_argument(
        "--atom-type-diffusion", choices=("mask", "uniform"), default="mask",
        help="物种扩散类型：mask=吸收型（默认，随机性小）/ uniform=均匀型",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = "cpu"

    crystal = load_crystal(args.structure)
    _, index0, coords0, lattice0 = to_tensors(crystal)
    n = crystal.num_atoms
    k = len(ELEMENT_VOCAB)
    mode = args.atom_type_diffusion
    print(f"结构: {n} 个原子, {k} 种元素; 物种走 D3PM({mode}), 坐标/晶格走连续扩散")

    batch_i = index0.unsqueeze(0).expand(args.batch_size, -1).contiguous()
    batch_c = coords0.unsqueeze(0).expand(args.batch_size, -1, -1).contiguous()
    batch_l = lattice0.unsqueeze(0).expand(args.batch_size, -1, -1).contiguous()

    model = CrystalDenoiser(
        n, k, num_species_input=num_species_input(k, mode), hidden=args.hidden, num_layers=args.layers
    )
    print("模型参数量:", sum(p.numel() for p in model.parameters()))
    schedule = CosineSchedule(args.num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = (args.species_weight, 1.0, 1.0)

    for step in range(args.steps):
        optimizer.zero_grad()
        loss, parts = training_loss(model, schedule, batch_i, batch_c, batch_l, mode, weights)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 200 == 0:
            print(
                f"update {step + 1:5d}  loss={loss.item():.5f}  "
                f"(物种 CE {parts['species_ce']:.5f}  坐标 {parts['coords_mse']:.5f}  晶格 {parts['lattice_mse']:.5f})"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "steps": args.steps, "num_steps": args.num_steps},
        args.output_dir / "crystal_d3pm.pt",
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    gen_i, gen_c, gen_l = sample(
        model, schedule, args.samples, n, k, device, generator,
        species_argmax=args.species_argmax, mode=mode,
    )

    print("\n生成结果（对比真实结构）:")
    species_ok = 0
    for i in range(args.samples):
        generated = decode(gen_i[i], gen_c[i], gen_l[i])
        write_poscar(
            args.output_dir / f"sample_{i:04d}.vasp",
            generated.lattice,
            generated.species,
            generated.frac_coords,
            comment=f"Generated sample {i}",
        )
        # 坐标误差必须用最小镜像：分数坐标里 0 和 1 是同一个位置，
        # 直接用 |gen - target| 会把「差一个周期」错算成 ~1.0 的误差。
        diff = generated.frac_coords - crystal.frac_coords
        diff = diff - np.round(diff)
        coord_err = float(np.abs(diff).mean())
        cart_err = float(np.linalg.norm(diff @ crystal.lattice, axis=-1).mean())
        lat_err = float(abs(generated.lattice - crystal.lattice).mean())
        match = sum(a == b for a, b in zip(generated.species, crystal.species))
        composition_ok = sorted(generated.species) == sorted(crystal.species)
        species_ok += int(composition_ok)
        print(
            f"sample {i}: 物种逐位匹配 {match}/{n}  组成一致 {composition_ok}  "
            f"坐标 MAE(周期感知) {coord_err:.4f} ({cart_err:.3f} 埃)  晶格 MAE {lat_err:.4f} 埃"
        )
    print(f"物种组成正确: {species_ok}/{args.samples}")

    print(f"\n模型和 {args.samples} 个 POSCAR 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
