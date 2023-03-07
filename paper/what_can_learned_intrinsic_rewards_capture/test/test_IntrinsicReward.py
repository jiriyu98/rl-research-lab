import unittest
import torch
import numpy as np

from paper.what_can_learned_intrinsic_rewards_capture.models.IntrinsicReward import IntrinsicReward


class TestIntrinsicReward(unittest.TestCase):
    def test_intrinsicreward_shape_without_hidden_state(self):
        intrinsicReward = IntrinsicReward(state_space=2)

        input = (np.random.randn(2,),
                 np.random.randn(1,),
                 np.random.randn(1,),
                 np.random.randn(1,),)

        sn, hn = intrinsicReward(input)

        self.assertEqual(torch.Size((1, 1)), sn.shape)
        self.assertEqual(torch.Size((1, 1)), hn.shape)

    def test_intrinsicreward_shape_with_hidden_state(self):
        intrinsicReward = IntrinsicReward(state_space=2)

        input = (np.random.randn(2,),
                 np.random.randn(1,),
                 np.random.randn(1,),
                 np.random.randn(1,),)

        s0, h0 = intrinsicReward(input)
        sn, hn = intrinsicReward(input, h0)

        self.assertEqual(torch.Size((1, 1)), sn.shape)
        self.assertEqual(torch.Size((1, 1)), hn.shape)
