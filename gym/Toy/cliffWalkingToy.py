import gym
import numpy as np
import matplotlib.pyplot as plt

from gym.utils.env_checker import check_env
from os.path import exists
from gym.core import Env
from typing import Tuple

class CliffWalkingQLearningAgent():
    def __init__(self, env: Env) -> None:
        """
        Args:
            q_array (np.arrays(a, b)): arbitrarily except that Q(terminal, *) = 0
        """
        self.alpha = 0.1 # step size
        self.epsilon = 0.1 # small epsilon
        self.gamma = 0.9
        self.observation_shape = env.observation_space.n
        self.action_shape = env.action_space.n
        self.q_array = np.zeros((self.observation_shape, self.action_shape))
        self.env = env
        self.q_array_file_name = "./cliffWalkingQLearningQTable.npy"
        self.reward = 0
        self.name = "QLearning"

    def epsilonGreedy(self, observation) -> int:
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            Q = self.q_array[observation, :]
            return np.random.choice(np.where(Q == np.max(Q))[0])
        else:
            return np.random.choice(self.action_shape)

    def step(self, observation) -> Tuple[int, int]:
        action = self.epsilonGreedy(observation)
        Q = self.q_array

        next_observation, reward, terminated, _, _ = self.env.step(action)
        self.reward += reward
        if not terminated:
            Q[observation, action] = Q[observation, action] + self.alpha * (reward + self.gamma * np.max(Q[next_observation, :] - Q[observation, action]))

        return next_observation, terminated
    
    def learn(self):
        self.reward = 0
        observation, _ = self.env.reset()
        terminated = False

        self.step_count = 0
        while not terminated:
            self.step_count += 1
            observation, terminated = self.step(observation)

    def saveQArray(self) -> None:
        np.save(self.q_array_file_name, self.q_array)

    def loadQArray(self) -> None:
        if exists(self.q_array_file_name):
            self.q_array = np.load(self.q_array_file_name)

class CliffWalkingSarsaAgent():
    def __init__(self, env: Env) -> None:
        """
        Args:
            q_array (np.arrays(a, b)): arbitrarily except that Q(terminal, *) = 0
        """
        self.alpha = 0.1 # step size
        self.epsilon = 0.1 # small epsilon
        self.gamma = 0.9
        self.observation_shape = env.observation_space.n
        self.action_shape = env.action_space.n
        self.q_array = np.zeros((self.observation_shape, self.action_shape))
        self.env = env
        self.q_array_file_name = "./cliffWalkingSarsaQTable.npy"
        self.reward = 0
        self.name = "Sarsa"

    def epsilonGreedy(self, observation) -> int:
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            Q = self.q_array[observation, :]
            return np.random.choice(np.where(Q == np.max(Q))[0])
        else:
            return np.random.choice(self.action_shape)

    def step(self, action, observation) -> Tuple[int, int]:
        Q = self.q_array

        next_observation, reward, terminated, _, _ = self.env.step(action)
        next_action = self.epsilonGreedy(next_observation)
        self.reward += reward
        if not terminated:
            Q[observation, action] = Q[observation, action] + self.alpha * (reward + self.gamma * Q[next_observation, next_action] - Q[observation, action])

        return next_observation, next_action, terminated
    
    def learn(self):
        self.reward = 0
        observation, _ = self.env.reset()
        action = self.epsilonGreedy(observation)
        terminated = False

        self.step_count = 0
        while not terminated:
            self.step_count += 1
            observation, action, terminated = self.step(action, observation)

    def saveQArray(self) -> None:
        np.save(self.q_array_file_name, self.q_array)

    def loadQArray(self) -> None:
        if exists(self.q_array_file_name):
            self.q_array = np.load(self.q_array_file_name)

# observation, info = env.reset(seed=42)
env1 = gym.make('CliffWalking-v0')
env2 = gym.make('CliffWalking-v0')
env2 = gym.make('CliffWalking-v0', render_mode="human")
check_env(env1.unwrapped)
check_env(env2.unwrapped)

cliffWalkingSarsaAgent = CliffWalkingSarsaAgent(env1)
cliffWalkingQLearningAgent = CliffWalkingQLearningAgent(env2)

fig = plt.figure()
plt.xlabel("Episodes")
plt.ylabel("rewards")
plt.title("Average Rewards - Episodes")

episodes_num = 2000

# cliffWalkingQLearningAgent.loadQArray()

def train(cliffWalkingAgent):
    rewards_array = []
    average_reward_array = []
    average_reward = 0
    for i in range(episodes_num):
        cliffWalkingAgent.learn()
        
        reward = cliffWalkingAgent.reward
        rewards_array.append(reward)
        print("Episodes {}, steps: {}, Rewards: {}".format(i, cliffWalkingAgent.step_count, reward))
        average_reward = (average_reward * i + reward) / (i + 1)
        average_reward_array.append(average_reward)

    plt.plot(np.arange(episodes_num), np.array(rewards_array), label="{} reward".format(cliffWalkingAgent.name), marker='.', markersize=1, linestyle='None')
    plt.plot(np.arange(episodes_num), np.array(average_reward_array), label="{} reward average".format(cliffWalkingAgent.name))

    cliffWalkingAgent.saveQArray()

train(cliffWalkingQLearningAgent)
# train(cliffWalkingSarsaAgent)
# EMA moving average
# fixed sliding window moving average
plt.plot(np.arange(episodes_num), [-13] * episodes_num)
plt.legend()
plt.show()
