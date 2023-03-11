import unittest
import torch
import numpy as np

from paper.what_can_learned_intrinsic_rewards_capture.models.IntrinsicReward import IntrinsicReward


class TestIntrinsicReward(unittest.TestCase):
    def test_intrinsicreward_shape_without_hidden_state(self):
        intrinsicReward = IntrinsicReward(input_space=5)

        input = (np.random.randn(2,),
                 np.random.randint(100),
                 np.random.randint(100),
                 np.random.randint(100),)

        sn, hn = intrinsicReward(input)

        self.assertEqual(torch.Size((1, 1)), sn.shape)
        self.assertEqual(torch.Size((1, 64)), hn.shape)

    def test_intrinsicreward_shape_with_hidden_state(self):
        intrinsicReward = IntrinsicReward(input_space=5)

        input = (np.random.randn(2,),
                 np.random.randint(100),
                 np.random.randint(100),
                 np.random.randint(100),)

        s0, h0 = intrinsicReward(input)
        sn, hn = intrinsicReward(input, h0)

        self.assertEqual(torch.Size((1, 1)), sn.shape)
        self.assertEqual(torch.Size((1, 64)), hn.shape)
