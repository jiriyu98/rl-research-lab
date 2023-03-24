import gym

import numpy as np


class Bandits(gym.Env):
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
