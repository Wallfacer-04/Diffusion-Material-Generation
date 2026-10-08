# Continuous-token DDPM

`schedule.py` owns the shared cosine schedule (100 steps by default).
`forward.py` adds noise and computes the training loss; `reverse.py` samples
with fixed posterior variance. The denoiser lives in `src/models/dit.py`.

With `single-stru` on Python's import path:

```python
import torch
from src.diffusion import DiffusionSchedule, ForwardDiffusion, ReverseDiffusion
from src.models.dit import DiTDenoiser
from src.utils.embedding_direct import DirectEmbed
from src.utils.token_scaler import TokenScaler

embedder = DirectEmbed()
tokens = embedder.encode(*embedder.read_structure("single-stru/data/Ba2PrCu3O7.json"))
scaler = TokenScaler.fit(tokens)
x0 = scaler.transform(tokens).unsqueeze(0)

model = DiTDenoiser(num_tokens=tokens.shape[0])  # width 32, 4 heads, 2 blocks
schedule = DiffusionSchedule(num_steps=100)
forward = ForwardDiffusion(schedule)
reverse = ReverseDiffusion(schedule)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer.zero_grad()
loss = forward.training_loss(model, x0)
loss.backward()
optimizer.step()  # Repeat training updates before expecting useful samples.

scaled_samples = reverse.sample(
    model, (4, tokens.shape[0], 3), torch.Generator().manual_seed(42)
)
continuous_tokens = scaler.inverse_transform(scaled_samples)
```

The forward equation is `xt = sqrt(alpha_bar[t])*x0 + sqrt(1-alpha_bar[t])*noise`.
Training uses uniformly sampled integer timesteps and equal per-element noise
MSE. The sampler predicts x0 from the noise estimate, then uses the analytical
posterior mean and variance. No clipping is applied. Timesteps are zero-based:
index 0 is the first noisy level, and its reverse step adds no random noise.

Diffusion inputs and outputs stay `(batch, tokens, 3)`. The hidden projection
exists only inside the denoiser. Atom count and token ordering are fixed;
position embeddings distinguish rows. Scaling is fitted once on training data:
atom digits use `A/2-1`, coordinates use `2*X-1`, and lattice vectors use
`L/max(1, max(abs(training_L)))`. Save the scaler's `state_dict()` with the model;
do not fit it again on generated samples. Generated atom values remain continuous
and require a separate discretization policy before `DirectEmbed.decode()`.

The model, schedule, scaler, and input tensors must share a device. Move modules
with `.to(device)` before use. Full sampling temporarily switches the model to
evaluation mode, disables gradients, and restores its previous modes. Standalone
`p_sample()` disables gradients but leaves evaluation mode to the caller.

From the repository root in the `diffusion` environment:

```text
python single-stru/scripts/test/diffusion/diffusion_test.py
python single-stru/scripts/diffusion_demo.py --updates 500
```

The demo defaults to CPU, seed 0, batch size 16, four generated samples, and no
saved files. `--hidden-size`, `--num-steps`, and `--updates` are configurable.
An optional `--output-dir single-stru/outputs/demo` saves the model, scaler,
schedule, training tokens, and continuous samples together. Outputs must stay
outside `src/`.

The initial 500-update run reduced fixed-evaluation noise MSE from 1.009277 to
0.012758, but sample reconstruction was poor: atom-digit RMSE ranged from
176.57 to 248.66, fractional-coordinate RMSE from 76.03 to 177.08, and lattice
RMSE from 1061.71 to 2192.81 angstroms. All values and gradients were finite.
This is a runnable baseline, not a demonstrated structure reconstruction model.
Near the terminal cosine timestep, a small noise-prediction error can become a
large clean-token error; average noise MSE alone does not establish sample quality.
