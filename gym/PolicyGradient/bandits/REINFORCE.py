import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical


# env
state_size = 1
action_size = 100

np.random.seed(0)

# multibandts -> one-state Markov decison process
# https://en.wikipedia.org/wiki/Multi-armed_bandit


class MultiarmedBanditsEnv():
    def __init__(self, nr_arms=10):
        self.action_space = gym.spaces.Discrete(nr_arms)
        self.observation_space = gym.spaces.Discrete(1)
        self.state = np.array([0])
        self.optimal_value = 0

    def step(self, action):
        assert self.action_space.contains(action)
        reward = (np.random.randn(1) + self.values[action]).item()

        return self.state, reward, False, {self.optimal}, ""

    def reset(self):
        self.values = np.random.randn(self.action_space.n)
        self.optimal = np.argmax(self.values)
        self.optimal_value = np.max(self.values)
        print("This reset will lead to the max return should be %f" %
              np.max(self.values))
        return self.state

    def render(self, mode='human', close=False):
        print("You are playing a %d-armed bandit" % self.action_space.n)


env = MultiarmedBanditsEnv(nr_arms=action_size)
torch.manual_seed(0)


# agent

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_size, 156)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(156, action_size)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state.flatten()).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs = m.log_prob(action)
    return action.item()


def finish_step(reward, gamma=1.0):
    returns = torch.tensor(reward)

    policy_loss = -policy.saved_log_probs * returns

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


def main():
    for i_episode in range(1):
        state = env.reset()
        step_reward = 0
        running_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            if False:
                env.render()
            step_reward = reward

            if done:
                break
            finish_step(reward)

            running_reward = 0.05 * step_reward + (1 - 0.05) * running_reward

            if t % 10 == 0:
                print('step {:5d}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tMax return: {:.2f}'.format(
                    t, step_reward, running_reward, env.optimal_value))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()
