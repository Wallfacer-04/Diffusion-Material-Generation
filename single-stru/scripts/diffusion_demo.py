"""Train a tiny CPU DiT on one structure and inspect continuous DDPM samples.

Run: python single-stru/scripts/diffusion_demo.py --updates 500
Optional --output-dir saves continuous tensors and reusable model/scaler state.
No atomic discretization or material-validity assessment is performed.
"""

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion import DiffusionSchedule, ForwardDiffusion, ReverseDiffusion
from src.models.dit import DiTDenoiser
from src.utils.embedding_direct import DirectEmbed
from src.utils.token_scaler import TokenScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=PROJECT_ROOT / "data/Ba2PrCu3O7.json")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1:
        parser.error("updates and batch-size must be positive")
    if args.output_dir:
        output_dir = args.output_dir.resolve()
        if output_dir == PROJECT_ROOT / "src" or PROJECT_ROOT / "src" in output_dir.parents:
            parser.error("output-dir must be outside src/")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    embedder = DirectEmbed()
    tokens = embedder.encode(*embedder.read_structure(args.structure))
    scaler = TokenScaler.fit(tokens)
    x0 = scaler.transform(tokens).unsqueeze(0).expand(args.batch_size, -1, -1)
    model = DiTDenoiser(tokens.shape[0], hidden_size=args.hidden_size)
    schedule = DiffusionSchedule(args.num_steps)
    forward = ForwardDiffusion(schedule)
    reverse = ReverseDiffusion(schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Use a fixed independent set of timesteps/noises to compare before and after.
    evaluation_rng = torch.Generator().manual_seed(args.seed + 1)
    eval_x0 = x0[:1].expand(128, -1, -1)
    eval_t = torch.randint(args.num_steps, (128,), generator=evaluation_rng)
    eval_noise = torch.randn(eval_x0.shape, generator=evaluation_rng)

    def evaluation_loss():
        with torch.no_grad():
            return forward.training_loss(model, eval_x0, eval_t, eval_noise).item()

    initial_loss = evaluation_loss()
    print(f"Atoms: {scaler.num_atoms}; tokens: {tokens.shape[0]}; hidden size: {args.hidden_size}")
    print(f"CPU training: {args.updates} updates, batch size {args.batch_size}, T={args.num_steps}")
    print(f"Initial evaluation noise MSE: {initial_loss:.6f}", flush=True)
    for step in range(args.updates):
        optimizer.zero_grad()
        loss = forward.training_loss(model, x0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at update {step + 1}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"Nonfinite gradient at update {step + 1}")
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.updates:
            print(f"Update {step + 1}: training noise MSE={loss.item():.6f}", flush=True)

    final_loss = evaluation_loss()
    if not torch.isfinite(torch.tensor([initial_loss, final_loss])).all():
        raise RuntimeError("Nonfinite evaluation loss")
    sampled = reverse.sample(
        model, (4, tokens.shape[0], 3), torch.Generator().manual_seed(args.seed + 2)
    )
    raw_samples = scaler.inverse_transform(sampled)
    if not torch.isfinite(raw_samples).all():
        raise RuntimeError("Generated tokens contain nonfinite values")
    n = scaler.num_atoms
    print(f"Final evaluation noise MSE: {final_loss:.6f}")
    for label, group in (("Atom digits", slice(0, n)), ("Fractional coordinates", slice(n, 2*n)),
                         ("Lattice (angstrom)", slice(2*n, None))):
        error = raw_samples[:, group] - tokens[group]
        per_sample_rmse = error.square().mean(dim=(1, 2)).sqrt()
        print(f"{label} RMSE per generated sample: {[round(v, 6) for v in per_sample_rmse.tolist()]}")
    print("Samples are continuous tokens; no atomic rounding, wrapping, or structure decoding applied.")
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_config": {"num_tokens": tokens.shape[0], "hidden_size": args.hidden_size},
            "model": model.state_dict(), "scaler": scaler.state_dict(),
            "num_steps": args.num_steps, "schedule": schedule.state_dict(),
            "seed": args.seed, "updates": args.updates,
            "training_tokens": tokens, "samples": raw_samples,
        }, output_dir / "diffusion_demo.pt")
        print(f"Saved tensors and checkpoint to {output_dir / 'diffusion_demo.pt'}")


if __name__ == "__main__":
    main()
