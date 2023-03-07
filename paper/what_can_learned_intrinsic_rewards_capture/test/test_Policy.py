import unittest
import torch
import numpy as np

from paper.what_can_learned_intrinsic_rewards_capture.models.Policy import Policy


class TestPolicy(unittest.TestCase):
    def test_policy_shape(self):
        policy = Policy(state_space=2, action_space=4)

        x = np.random.randn(2,)
        y, _ = policy(x)

        self.assertEqual(torch.Size((1, )), y.shape)
