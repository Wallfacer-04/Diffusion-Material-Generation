"""CPU checks: python single-stru/scripts/test/diffusion/diffusion_test.py."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.diffusion import DiffusionSchedule, ForwardDiffusion, ReverseDiffusion
from src.models.dit import DiTDenoiser
from src.utils.token_scaler import TokenScaler


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.schedule = DiffusionSchedule(20)
        self.forward = ForwardDiffusion(self.schedule)
        self.reverse = ReverseDiffusion(self.schedule)
        self.x0 = torch.randn(3, 7, 3)
        self.t = torch.tensor([0, 9, 19])
        self.noise = torch.randn_like(self.x0)

    def test_scaling_roundtrip_and_saved_state(self):
        tokens = self.x0.clone()
        tokens[:, -3:] *= 8
        scaler = TokenScaler.fit(tokens)
        self.assertEqual(scaler.num_atoms, 2)
        self.assertAlmostEqual(scaler.lattice_scale.item(), tokens[:, -3:].abs().max().item())
        torch.testing.assert_close(scaler.inverse_transform(scaler.transform(tokens)), tokens)
        restored = TokenScaler(1, 1)
        restored.load_state_dict(scaler.state_dict())
        torch.testing.assert_close(restored.transform(tokens), scaler.transform(tokens))
        # Inversion also works for generated values outside the training ranges.
        generated = 5 * torch.randn_like(tokens)
        torch.testing.assert_close(scaler.transform(scaler.inverse_transform(generated)), generated)

    def test_schedule_and_forward_formula(self):
        s = self.schedule
        self.assertTrue(torch.all((s.betas > 0) & (s.betas < 1)))
        self.assertTrue(torch.all(s.alpha_bar[1:] < s.alpha_bar[:-1]))
        self.assertLess(s.alpha_bar[-1].item(), 1e-4)
        self.assertEqual(s.posterior_variance[0].item(), 0)
        ab = (1 - s.betas.double()).cumprod(0)[self.t, None, None]
        expected = ab.sqrt() * self.x0 + (1 - ab).sqrt() * self.noise
        actual = self.forward.q_sample(self.x0, self.t, self.noise)
        torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)

    def test_true_noise_recovers_posterior(self):
        noise = self.noise

        class Oracle(nn.Module):
            def forward(self, x, t):
                return noise

        s = self.schedule
        xt = self.forward.q_sample(self.x0, self.t, noise)
        mean, variance = self.reverse.p_mean_variance(Oracle(), xt, self.t)
        beta = s.betas.double()[self.t, None, None]
        alpha = 1 - beta
        bars = (1 - s.betas.double()).cumprod(0)
        previous = torch.cat((torch.ones(1, dtype=torch.double), bars[:-1]))
        ab = bars[self.t, None, None]
        prev = previous[self.t, None, None]
        expected_mean = beta * prev.sqrt() / (1 - ab) * self.x0
        expected_mean += alpha.sqrt() * (1 - prev) / (1 - ab) * xt
        expected_variance = beta * (1 - prev) / (1 - ab)
        torch.testing.assert_close(mean.double(), expected_mean, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(variance.double(), expected_variance, atol=1e-6, rtol=1e-6)
        sample = self.reverse.p_sample(Oracle(), xt, self.t)
        torch.testing.assert_close(sample[0], self.x0[0])
        self.assertEqual(self.forward.training_loss(Oracle(), self.x0, self.t, noise).item(), 0)

    def test_last_step_does_not_consume_randomness(self):
        model = DiTDenoiser(7)
        t = torch.zeros(3, dtype=torch.long)
        generator = torch.Generator().manual_seed(11)
        state = generator.get_state().clone()
        sample = self.reverse.p_sample(model, self.x0, t, generator)
        torch.testing.assert_close(generator.get_state(), state)
        torch.testing.assert_close(sample, self.reverse.p_mean_variance(model, self.x0, t)[0])

    def test_full_chain_with_analytic_single_structure_denoiser(self):
        schedule = DiffusionSchedule(100)
        x0 = self.x0[:1]

        class SingleStructureOracle(nn.Module):
            def forward(self, xt, t):
                signal = schedule.extract(schedule.sqrt_alpha_bar, t, xt)
                noise_scale = schedule.extract(schedule.sqrt_one_minus_alpha_bar, t, xt)
                return (xt - signal * x0) / noise_scale

        samples = ReverseDiffusion(schedule).sample(
            SingleStructureOracle(), (4, 7, 3), torch.Generator().manual_seed(8)
        )
        torch.testing.assert_close(samples, x0.expand_as(samples), atol=2e-6, rtol=2e-6)

    def test_training_and_sampling(self):
        model = DiTDenoiser(7)
        prediction = model(self.x0, self.t)
        self.assertEqual(prediction.shape, self.x0.shape)
        self.assertEqual(prediction.count_nonzero().item(), 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            loss = self.forward.training_loss(model, self.x0)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertGreater(model.blocks[0].attn.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.time_mlp[0].weight.grad.abs().sum().item(), 0)
        model.train()
        model.blocks[0].eval()
        flags = [module.training for module in model.modules()]
        sample1 = self.reverse.sample(model, (2, 7, 3), torch.Generator().manual_seed(42))
        sample2 = self.reverse.sample(model, (2, 7, 3), torch.Generator().manual_seed(42))
        self.assertEqual(flags, [module.training for module in model.modules()])
        self.assertFalse(sample1.requires_grad)
        self.assertTrue(torch.isfinite(sample1).all())
        torch.testing.assert_close(sample1, sample2, rtol=0, atol=0)

    def test_reject_invalid_inputs_and_restore_mode_on_error(self):
        with self.assertRaises(ValueError):
            self.forward.q_sample(self.x0, torch.tensor([0, 1, 20]))
        with self.assertRaises(ValueError):
            self.forward.q_sample(self.x0, self.t, torch.randn(1, 7, 3))
        model = DiTDenoiser(7).train()
        with self.assertRaises(ValueError):
            self.reverse.sample(model, (1, 8, 3))
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
