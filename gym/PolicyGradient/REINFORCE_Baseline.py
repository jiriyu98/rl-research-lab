import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.distributions import Categorical


# env
env = gym.make('CartPole-v1')
env.reset(seed=0)
torch.manual_seed(0)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 256)
        self.affine3 = nn.Linear(256, 2)

        self.pairs = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=-1)


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
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()

# state-value function
stateValue = StateValue()
optimizer_stateValue = optim.Adam(stateValue.parameters(), lr=1e-3)


def get_probs(state, action):
    probs = policy(state)
    m = Categorical(probs)
    return m.log_prob(action)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.pairs.append((state, action))
    return action.item()

# update value function (w) and policy (theta)


def update(rewards, gamma=1.0):
    # policy update
    R = 0
    returns = []

    rewards = rewards[::-1]

    for r in rewards:
        R = r + gamma * R
        returns.append(R)
    returns = returns[::-1]
    returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)

    for i, ((state, action), R) in enumerate(zip(policy.pairs, returns)):
        # See these posts:
        # https://github.com/pytorch/pytorch/issues/39141
        # https://discuss.pytorch.org/t/solved-pytorch1-5-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/90256/2
        '''
            This is most likely happening because value_optimizer.step() actually 
            modifies the weights of the model inplace while the original value of 
            these weights is needed to compute action_loss.backward(). 
        '''

        # REINFORCE with Baseline
        delta = R - stateValue(state)

        optimizer_stateValue.zero_grad()
        loss = -stateValue(state) * delta.detach()
        loss.backward()
        optimizer_stateValue.step()

        optimizer.zero_grad()
        loss = -get_probs(state, action) * delta.detach() * (gamma ** i)
        loss.backward()
        optimizer.step()

        # # REINFORCE without Baseline
        # # high variance
        # optimizer.zero_grad()
        # loss = -get_probs(state, action) * (gamma ** i)
        # loss.backward()
        # optimizer.step()

    # del
    del policy.pairs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        rewards = []

        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            rewards.append(reward)

            ep_reward += reward

            if done:
                break

        update(rewards)
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(np.arange(1, len(scores)+1), scores)
        # plt.ylabel('Score')
        # plt.xlabel('Episode #')
        # plt.show()


if __name__ == '__main__':
    main()
