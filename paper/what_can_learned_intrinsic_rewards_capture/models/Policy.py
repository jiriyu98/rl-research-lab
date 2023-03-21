import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    # CNN

    # ref: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
    def __init__(self, state_space, action_space):
        super().__init__()
        self.affine1 = nn.Linear(state_space, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.action_head = nn.Linear(128, action_space)
        self.value_head = nn.Linear(128, 1)

    def update_parameters(self):
        None

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        x = F.relu(self.dropout(self.affine1(x)))

        action_prob = F.softmax(self.action_head(x), dim=1)
        state_value = self.value_head(x)
        # m = Categorical(action_prob)
        # action = m.sample()

        return action_prob, state_value
