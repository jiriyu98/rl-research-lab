import torch
import torch.nn as nn
import torch.nn.functional as F

# different


# one possible way
# [s0, a0, r1, d1, s1, a1, ..., rn, dn, sn]
# 4-tuple, (s0, a0, r1, d1), ..., (sn-1, an-1, rn, dn)
# 4-layer rnn (i.e. stack 4 RNNs)
# How to deal: (sn, 0, 0, 0) -> (output, _, _, _)
class IntrinsicReward(nn.Module):
    def __init__(self, input_space):
        super().__init__()
        self.rnn_hidden_size = 64

        self.rnn = nn.RNN(input_space, self.rnn_hidden_size)
        self.affline1 = nn.Linear(self.rnn_hidden_size, 128)
        self.affline2 = nn.Linear(128, 1)

    def forward(self, input, hidden_state=None):
        s0, a0, r1, d1 = input

        s0 = torch.from_numpy(s0).float().unsqueeze(0).detach()
        a0 = torch.tensor([a0]).unsqueeze(0).detach()
        r1 = torch.tensor([r1]).unsqueeze(0).detach()
        d1 = torch.tensor([d1]).unsqueeze(0).detach()

        # cat all the input
        x = torch.cat((s0, a0, r1, d1), 1)

        if hidden_state is None:
            h0 = self.initHidden()
        else:
            h0 = hidden_state

        h0 = h0.detach()

        x, hn = self.rnn(x, h0)
        x = F.relu(self.affline1(x))
        x = F.relu(self.affline2(x))

        # intrinsicreward and history info
        return x, hn

    def initHidden(self):
        return torch.zeros((1, self.rnn_hidden_size))
