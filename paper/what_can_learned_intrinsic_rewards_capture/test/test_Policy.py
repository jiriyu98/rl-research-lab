import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from paper.what_can_learned_intrinsic_rewards_capture.models.Policy import \
    Policy


class TestPolicy(unittest.TestCase):
    def test_policy_shape(self):
        policy = Policy(state_space=2, action_space=4)

        x = np.random.randn(2,)
        y1, y2 = policy(x)

        self.assertEqual(torch.Size((1, 4)), y1.shape)
        self.assertEqual(torch.Size((1, 1)), y2.shape)

    def test_policy_on_small_examples(self):
        def selectAction(self, observation):
            action_prob, _ = self.policy(observation["agent"])
            m = Categorical(action_prob)
            action = m.sample()
            return action.item()
