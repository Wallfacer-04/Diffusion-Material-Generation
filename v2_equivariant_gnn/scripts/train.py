"""在单个晶体结构上训练 v2（等变 GNN + 不变晶格表示），再从噪声生成它自己。

用法：
    python 14_晶体扩散等变GNN/scripts/train.py --steps 4000 --samples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gram import gram6_from_lattice
from src.data.structure import ELEMENT_VOCAB, GRAM_SCALE, decode, load_crystal, to_tensors
from src.diffusion.forward import training_loss
from src.diffusion.reverse import sample
from src.diffusion.schedule import CosineSchedule
from src.models.egnn import CrystalEGNN
from src.utils.poscar import write_poscar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=PROJECT_ROOT / "data" / "Ba2PrCu3O7.json")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=200, help="扩散总步数 T")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--no-slot", action="store_true", help="去掉槽位嵌入（置换等变，但单结构重建会明显变差）")
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
    species0, coords0, gram0 = to_tensors(crystal)
    n = crystal.num_atoms
    k = len(ELEMENT_VOCAB)
    print(f"结构: {n} 个原子, {k} 种元素")

    batch_s = species0.unsqueeze(0).expand(args.batch_size, -1, -1).contiguous()
    batch_c = coords0.unsqueeze(0).expand(args.batch_size, -1, -1).contiguous()
    batch_g = gram0.unsqueeze(0).expand(args.batch_size, -1).contiguous()

    model = CrystalEGNN(n, k, hidden=args.hidden, num_layers=args.layers, use_slot=not args.no_slot)
    print("模型参数量:", sum(p.numel() for p in model.parameters()))
    print("槽位嵌入:", "开（默认，保证单结构重建）" if not args.no_slot else "关（置换等变）")
    schedule = CosineSchedule(args.num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        optimizer.zero_grad()
        loss, parts = training_loss(model, schedule, batch_s, batch_c, batch_g)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 200 == 0:
            print(
                f"update {step + 1:5d}  loss={loss.item():.5f}  "
                f"(物种 {parts['species']:.5f}  坐标 {parts['coords']:.5f}  Gram {parts['gram']:.5f})"
            )

    print(f"\n相对位移向量场的门控 coord_gate = {model.coord_gate.item():.4f}（0 = 未使用该通路）")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "steps": args.steps, "num_steps": args.num_steps},
        args.output_dir / "crystal_egnn.pt",
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    gen_s, gen_c, gen_g = sample(model, schedule, args.samples, n, k, device, generator)

    target_gram = gram6_from_lattice(crystal.lattice)
    print("\n生成结果（对比真实结构）:")
    for i in range(args.samples):
        generated = decode(gen_s[i], gen_c[i], gen_g[i])
        write_poscar(
            args.output_dir / f"sample_{i:04d}.vasp",
            generated.lattice,
            generated.species,
            generated.frac_coords,
            comment=f"Generated sample {i}",
        )
        coord_err = float(abs(generated.frac_coords - crystal.frac_coords).mean())
        gram_err = float(abs(gram6_from_lattice(generated.lattice) - target_gram).mean())
        lengths = [float((generated.lattice[j] ** 2).sum() ** 0.5) for j in range(3)]
        match = sum(a == b for a, b in zip(generated.species, crystal.species))
        print(
            f"sample {i}: 物种匹配 {match}/{n}  坐标 MAE {coord_err:.4f}  "
            f"Gram MAE {gram_err:.4f} 埃^2  晶格长度 {[round(v, 3) for v in lengths]}"
        )

    print(f"\n模型和 {args.samples} 个 POSCAR 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
