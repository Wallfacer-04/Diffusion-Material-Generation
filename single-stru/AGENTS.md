# Project instructions

## Purpose

The stage goal is a material generation model based on diffusion transformer, which uses only one structure as training set. We expect that the trained model will always generate the same structure.

## Environment

- Run `conda activate diffusion` before running Python commands.
- Use PyTorch for the diffusion model.
- Keep the first example runnable on a CPU without downloading data or weights.
- Avoid installing or upgrading packages unless the task needs it.

## Scope and verification

- Keep generated outputs separate from source code.
- Use a fixed random seed for repeatable demonstrations.
- Run the example in the `diffusion` environment after making changes.
- Check that the loss and generated values are finite, and inspect whether samples
  resemble the training distribution. Report what was actually run and observed.
- Do not claim that a toy model generates valid or stable materials.
