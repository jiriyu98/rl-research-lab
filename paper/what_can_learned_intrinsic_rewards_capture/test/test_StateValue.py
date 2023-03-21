import unittest
import torch
import numpy as np

from paper.what_can_learned_intrinsic_rewards_capture.models.StateValue import StateValue


class TestStateValue(unittest.TestCase):
    def test_stateValue_shape(self):
        stateValue = StateValue(state_space=2)

        x = np.random.randn(2,)
        y = stateValue(x)

        self.assertEqual(torch.Size((1, 1)), y.shape)
