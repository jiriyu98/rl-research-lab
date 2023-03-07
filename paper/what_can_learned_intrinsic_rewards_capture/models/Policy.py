import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    # CNN
    def __init__(self, state_space, action_space):
        super().__init__()
        self.affine1 = nn.Linear(state_space, 64)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(64, action_space)

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = F.softmax(x, dim=1)
        m = Categorical(x)

        action = m.sample()
        # save the log_prob

        return action, m.log_prob(action)
