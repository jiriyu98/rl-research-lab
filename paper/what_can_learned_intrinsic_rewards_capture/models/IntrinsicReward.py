import torch
import torch.nn as nn

# different


# one possible way
# [s0, a0, r1, d1, s1, a1, ..., rn, dn, sn]
# 4-tuple, (s0, a0, r1, d1), ..., (sn-1, an-1, rn, dn)
# 4-layer rnn (i.e. stack 4 RNNs)
# How to deal: (sn, 0, 0, 0) -> (output, _, _, _)
class IntrinsicReward(nn.Module):
    def __init__(self, state_space):
        super().__init__()

        self.rnn1 = nn.RNN(state_space, 1)
        self.rnn2 = nn.RNN(1, 1)
        self.rnn3 = nn.RNN(1, 1)
        self.rnn4 = nn.RNN(1, 1)

    def forward(self, input, hidden_state=None):
        s0, a0, r1, d1 = input

        s0 = torch.from_numpy(s0).float().unsqueeze(0)
        a0 = torch.from_numpy(a0).float().unsqueeze(0)
        r1 = torch.from_numpy(r1).float().unsqueeze(0)
        d1 = torch.from_numpy(d1).float().unsqueeze(0)

        if hidden_state is None:
            h0_1 = self.initHidden()
        else:
            h0_1 = hidden_state

        sn, hn_1 = self.rnn1(s0, h0_1)
        _, hn_2 = self.rnn2(a0, hn_1)
        _, hn_3 = self.rnn3(r1, hn_2)
        _, hn_4 = self.rnn4(d1, hn_3)

        # intrinsicreward and history info
        return sn, hn_4

    def initHidden(self):
        return torch.zeros((1, 1))
