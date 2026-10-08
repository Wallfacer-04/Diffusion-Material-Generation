"""Verify POSCAR round trips and invalid-output handling.

Run in the diffusion environment from the repository root:
    python single-stru/scripts/test/utils/outputs_to_poscars_test.py
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from pymatgen.core import Structure

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.embedding_direct import DirectEmbed
from src.utils.outputs_to_poscars import convert_outputs, samples_to_structures


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tokens = DirectEmbed().encode(*DirectEmbed.read_structure(PROJECT_ROOT / "data/Ba2PrCu3O7.json"))

    def test_checkpoint_round_trip_and_overwrite(self):
        n = (len(self.tokens) - 3) // 2
        shifted = self.tokens.clone()
        shifted[n:2*n] += 2
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "outputs.pt"
            torch.save({"samples": torch.stack((self.tokens, shifted))}, checkpoint)
            paths = convert_outputs(checkpoint, strict_atoms=True)
            self.assertEqual(len(paths), 2)
            expected_z, _, expected_lattice = DirectEmbed.decode(self.tokens)
            for path, tokens in zip(paths, (self.tokens, shifted)):
                structure = Structure.from_file(str(path))
                self.assertEqual(list(structure.atomic_numbers), expected_z.tolist())
                np.testing.assert_allclose(structure.lattice.matrix, expected_lattice.numpy(), atol=1e-7)
                np.testing.assert_allclose(structure.frac_coords, tokens[n:2*n].remainder(1).numpy(), atol=1e-7)
            with self.assertRaises(FileExistsError):
                convert_outputs(checkpoint)
            convert_outputs(checkpoint, overwrite=True)
            torch.save(self.tokens, checkpoint)
            self.assertEqual(len(convert_outputs(checkpoint, overwrite=True)), 1)

    def test_atom_projection(self):
        tokens = self.tokens.clone()
        tokens[0] = torch.tensor([0.1, 1.1, 3.1])
        self.assertEqual(samples_to_structures(tokens)[0].atomic_numbers[0], 8)
        with self.assertRaises(ValueError):
            samples_to_structures(tokens, strict_atoms=True)
        for digits in ([0., 0., 0.], [4., 4., 4.]):
            tokens[0] = torch.tensor(digits)
            self.assertTrue(1 <= samples_to_structures(tokens)[0].atomic_numbers[0] <= 118)
            with self.assertRaises(ValueError):
                samples_to_structures(tokens, strict_atoms=True)

    def test_invalid_samples_leave_no_output(self):
        for kind in ("nonfinite", "singular", "shape"):
            tokens = self.tokens.clone()
            if kind == "nonfinite":
                tokens[0, 0] = float("nan")
            elif kind == "singular":
                tokens[-1] = tokens[-2]
            else:
                tokens = tokens[:-1]
            with TemporaryDirectory() as folder:
                checkpoint = Path(folder) / "outputs.pt"
                torch.save({"samples": tokens}, checkpoint)
                with self.assertRaises(ValueError):
                    convert_outputs(checkpoint)
                self.assertFalse((Path(folder) / "poscars").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
