import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm

import matplotlib.pyplot as plt
from torch.distributions import Categorical


# env
env = gym.make('CartPole-v1')
env.reset(seed=0)
torch.manual_seed(0)

# no cude ^_^
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
https://www.reddit.com/r/reinforcementlearning/comments/don9ux/update_reinforce_algorithm_stepwise_or_episodewise/
'''

# class Policy(nn.Module):
#     def __init__(self):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(4, 128)
#         self.affine2 = nn.Linear(128, 256)
#         self.affine3 = nn.Linear(256, 2)
#         self.dropout = nn.Dropout(p=0.6)

#     def forward(self, x):
#         x = F.relu(self.affine1(x))
#         x = self.dropout(x)
#         x = F.relu(self.affine2(x))
#         x = self.affine3(x)
#         return F.softmax(x, dim=1)
    
# class StateValue(nn.Module):
#     def __init__(self):
#         super(StateValue, self).__init__()
#         self.affine1 = nn.Linear(4, 128)
#         self.affine2 = nn.Linear(128, 256)
#         self.affine3 = nn.Linear(256, 1)
#         self.dropout = nn.Dropout(p=0.6)

#     def forward(self, x):
#         x = F.relu(self.affine1(x))
#         x = self.dropout(x)
#         x = F.relu(self.affine2(x))
#         x = self.affine3(x)
#         return x

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
    
class StateValue(nn.Module):
    def __init__(self):
        super(StateValue, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 256)
        self.affine3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.affine3(x)
        return x

# policy
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

# state-value function
stateValue = StateValue()
optimizer_stateValue = optim.Adam(stateValue.parameters(), lr=1e-4)

def get_probs(state, action):
    probs = policy(state)
    m = Categorical(probs)
    return m.log_prob(action)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action

def update(state, next_state, action, reward, I, done, gamma=1.0):
    # transform to tensor
    state = torch.from_numpy(state).float().unsqueeze(0)
    next_state = torch.from_numpy(next_state).float().unsqueeze(0)

    if done:
        delta = reward - stateValue(state)
    else:
        delta = reward + gamma * stateValue(next_state) - stateValue(state)

    # update state-value function and policy
    optimizer_stateValue.zero_grad()
    optimizer.zero_grad()

    loss_critic = -stateValue(state) * delta.detach()
    loss_critic.backward()

    loss_actor = -get_probs(state, action) * delta.detach() * I
    loss_actor.backward()
    
    optimizer_stateValue.step()
    optimizer.step()

    return I * gamma

def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        I = 1

        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            next_state, reward, done, _, _ = env.step(action.item())
            ep_reward += reward

            # update, for simplicity, just update I here
            I = update(state, next_state, action, reward, I, done)
            state = next_state
            
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()