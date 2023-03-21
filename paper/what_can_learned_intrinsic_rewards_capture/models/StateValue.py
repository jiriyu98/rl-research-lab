import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class StateValue(nn.Module):
    # estimate V state-value function instead of Q action-value function
    def __init__(self, state_space):
        super().__init__()
        self.affine1 = nn.Linear(state_space, 64)
        self.affine2 = nn.Linear(64, 128)
        self.affine3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.affine3(x)
        return x
