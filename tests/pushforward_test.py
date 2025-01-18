import unittest
import torch
import numpy as np
from omegaconf import OmegaConf

from src.utils.train_utils import push_forward_sample_steps


class TestPushForward(unittest.TestCase):
    """Class for unit testing the push-forward functions."""

    def setUp(self):
        self.pf = OmegaConf.create(
            {
                "steps": [-1, 20000, 50000, 100000],
                "unrolls": [0, 1, 3, 20],
                "probs": [4.05, 4.05, 1.0, 1.0],
            }
        )

        self.seed = 42  # Set the seed for reproducibility

    def body_steps(self, step, expected_unrolls, expected_probs):
        # Normalize probabilities to ensure they sum to 1
        probs = np.array(expected_probs)
        probs = probs / probs.sum()

        # Collect the sampled unroll steps
        dump = []
        for _ in range(1000):
            self.seed, unroll_steps = push_forward_sample_steps(self.seed, step, self.pf)
            dump.append(unroll_steps)

        # Verify the unique unroll steps and their probabilities
        unique, counts = np.unique(dump, return_counts=True)
        self.assertTrue(
            (unique == expected_unrolls).all(),
            f"Expected unroll steps {expected_unrolls}, but got {unique}",
        )
        self.assertTrue(
            np.allclose(counts / 1000, probs, atol=0.05),
            f"Expected probabilities {probs}, but got {counts / 1000}",
        )

    def test_pf_step_1(self):
        self.body_steps(1, np.array([0]), np.array([1.0]))

    def test_pf_step_60000(self):
        self.body_steps(60000, np.array([0, 1, 3]), np.array([0.45, 0.45, 0.1]))


if __name__ == "__main__":
    unittest.main()
