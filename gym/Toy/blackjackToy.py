import gym
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from gym.utils.env_checker import check_env
from os.path import exists
from collections import defaultdict
from gym.core import Env
from typing import Tuple

# reason: to avoid the exploring starts, then it has on-policy and off-policy...
class BlackjackOnPolicyAgent():
    def __init__(self, env: Env, name) -> None:
        # observations_space: Tuple(Discrete(32), Discrete(11), Discrete(2))
        # action_space: Discrete(2)
        self.name = name
        self.env = env
        self.action_shape = env.action_space.n
        self.observation_space = env.observation_space
        self.Q = defaultdict(lambda : np.zeros(self.action_shape))
        self.Returns = defaultdict(lambda:(0.0, 0))
        self.pi = defaultdict(lambda : np.ones(self.action_shape) / self.action_shape) # for epsilon-soft policy
        self.gamma = 1.0
        self.epsilon = 0.1
    
    # epsilon greedy policy
    def epsilonGreedyPolicy(self, observation):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):
            Q = self.Q[observation]
            return np.random.choice(np.where(Q == np.max(Q))[0])
        else:
            return np.random.choice(self.action_shape)

    # epsilon soft policy
    def epsilonSoftPolicy(self, observation):
        return np.random.choice(self.action_shape, 1, p=self.pi[observation])[0]

    # Interaction with enviornment
    def step(self, obeservation):
        action = self.epsilonSoftPolicy(obeservation)

        next_observation, reward, terminated, _, _ = self.env.step(action)
        return next_observation, action, reward, terminated

    # Get episode and sort from end to begin
    def getEpisode(self):
        # [(S0, A0, R1), (S1, A1, R2), ..., (ST-1, AT-1, RT)]
        # T -> terminated time T
        episode = []
        observation, _ = self.env.reset()
        terminated = False

        while not terminated:
            next_observation, action, reward, terminated = self.step(observation)
            episode.append((observation, action, reward))
            observation = next_observation

        return episode

    def learnByFirstVisit(self):
        episode = self.getEpisode()

        T = len(episode)
        G = 0

        firstVisitSet = set()
        for i in range(T - 1, -1, -1):
            S, A, R = episode[i]
            G = self.gamma * G + R
            Q = self.Q
            Returns = self.Returns
            pi = self.pi

            if not (S, A) in firstVisitSet:
                firstVisitSet.add((S, A))

                sum, n = self.Returns[(S, A)]
                Returns[(S, A)] = (sum + G, n + 1)
                Q[S][A] = (sum + G) / (n + 1)
                AStar = np.random.choice(np.where(Q[S] == np.max(Q[S]))[0])

                for j in range(0, self.action_shape):
                    if j != AStar:
                        pi[S][j] = self.epsilon / self.action_shape
                    else:
                        pi[S][j] = 1 - self.epsilon + self.epsilon / self.action_shape

    def learnByEveryVisit(self):
            episode = self.getEpisode()

            T = len(episode)
            G = 0

            for i in range(T - 1, -1, -1):
                S, A, R = episode[i]
                G = self.gamma * G + R
                Q = self.Q
                Returns = self.Returns
                pi = self.pi

                sum, n = self.Returns[(S, A)]
                Returns[(S, A)] = (sum + G, n + 1)
                Q[S][A] = (sum + G) / (n + 1)
                AStar = np.random.choice(np.where(Q[S] == np.max(Q[S]))[0])

                for j in range(0, self.action_shape):
                    if j != AStar:
                        pi[S][j] = self.epsilon / self.action_shape
                    else:
                        pi[S][j] = 1 - self.epsilon + self.epsilon / self.action_shape

    def learn(self):
        self.learnByFirstVisit()

    def save(self, num):
        np.savez("blackjackdata/data_{}_{}.npz".format(self.name, num), Q=np.array(dict(self.Q)), pi=np.array(dict(self.pi)), Returns=np.array(dict(self.Returns)))

    def load(self, num):
        file_name = "blackjackdata/data_{}_{}.npz".format(self.name, num)
        if not exists(file_name):
            return False

        npzfile = np.load(file_name, allow_pickle=True)
        self.Q.update(npzfile['Q'].item())
        self.pi.update(npzfile['pi'].item())
        self.Returns.update(npzfile['Returns'].item())
        return True

class BlackjackOffPolicyAgent():
    def __init__(self, env: Env, name) -> None:
        # observations_space: Tuple(Discrete(32), Discrete(11), Discrete(2))
        # action_space: Discrete(2)
        self.name = name
        self.env = env
        self.action_shape = env.action_space.n
        self.observation_space = env.observation_space
        self.Q = defaultdict(lambda : np.zeros(self.action_shape))
        self.C = defaultdict(lambda : np.zeros(self.action_shape))
        self.pi = defaultdict(lambda : np.zeros(self.action_shape))
        self.b = defaultdict(lambda : np.ones(self.action_shape) / self.action_shape)
        self.gamma = 1.0
        self.epsilon = 0.1

    # equal probability soft policy b
    def RandomSoftPolicy(self, observation):
        return np.random.choice(self.action_shape, 1, p=self.b[observation])[0]

    # Interaction with enviornment
    def step(self, obeservation):
        action = self.RandomSoftPolicy(obeservation)

        next_observation, reward, terminated, _, _ = self.env.step(action)
        return next_observation, action, reward, terminated

    # Get episode and sort from end to begin
    def getEpisode(self):
        # [(S0, A0, R1), (S1, A1, R2), ..., (ST-1, AT-1, RT)]
        # T -> terminated time T
        episode = []
        observation, _ = self.env.reset()
        terminated = False

        while not terminated:
            next_observation, action, reward, terminated = self.step(observation)
            episode.append((observation, action, reward))
            observation = next_observation

        return episode

    def learn(self):
        episode = self.getEpisode()

        Q = self.Q
        C = self.C
        pi = self.pi
        T = len(episode)
        G = 0
        W = 1

        for i in range(T - 1, -1, -1):
            S, A, R = episode[i]
            G = self.gamma * G + R
            C[S][A] = C[S][A] + W
            Q[S][A] = Q[S][A] + W / C[S][A] * (G - Q[S][A])

            AStar = np.random.choice(np.where(Q[S] == np.max(Q[S]))[0])

            for j in range(self.action_shape):
                if j != AStar:
                    pi[S][j] = 0
                else:
                    pi[S][j] = 1

            if A != AStar:
                break
            W = W * 1 / self.b[S][A]


    def save(self, num):
        np.savez("blackjackdata/data_{}_{}.npz".format(self.name, num), Q=np.array(dict(self.Q)), C=np.array(dict(self.C)), pi=np.array(dict(self.pi)))

    def load(self, num):
        file_name = "blackjackdata/data_{}_{}.npz".format(self.name, num)
        if not exists(file_name):
            return False

        npzfile = np.load(file_name, allow_pickle=True)
        self.Q.update(npzfile['Q'].item())
        self.C.update(npzfile['C'].item())
        self.pi.update(npzfile['pi'].item())
        return True

class BlackjackOffPolicyAgentRandom():
    def __init__(self, env: Env, name) -> None:
        # observations_space: Tuple(Discrete(32), Discrete(11), Discrete(2))
        # action_space: Discrete(2)
        self.name = name
        self.env = env
        self.action_shape = env.action_space.n
        self.observation_space = env.observation_space
        self.Q = defaultdict(lambda : np.zeros(self.action_shape))
        self.C = defaultdict(lambda : np.zeros(self.action_shape))
        self.pi = defaultdict(lambda : np.zeros(self.action_shape))
        self.b = defaultdict(get_probability) # for epsilon-soft policy
        self.gamma = 1.0
        self.epsilon = 0.1

    # equal probability soft policy b
    def RandomSoftPolicy(self, observation):
        return np.random.choice(self.action_shape, 1, p=self.b[observation])[0]

    # Interaction with enviornment
    def step(self, obeservation):
        action = self.RandomSoftPolicy(obeservation)

        next_observation, reward, terminated, _, _ = self.env.step(action)
        return next_observation, action, reward, terminated

    # Get episode and sort from end to begin
    def getEpisode(self):
        # [(S0, A0, R1), (S1, A1, R2), ..., (ST-1, AT-1, RT)]
        # T -> terminated time T
        episode = []
        observation, _ = self.env.reset()
        terminated = False

        while not terminated:
            next_observation, action, reward, terminated = self.step(observation)
            episode.append((observation, action, reward))
            observation = next_observation

        return episode

    def learn(self):
        self.b = defaultdict(get_probability) # for epsilon-soft policy
        episode = self.getEpisode()

        Q = self.Q
        C = self.C
        pi = self.pi
        T = len(episode)
        G = 0
        W = 1

        for i in range(T - 1, -1, -1):
            S, A, R = episode[i]
            G = self.gamma * G + R
            C[S][A] = C[S][A] + W
            Q[S][A] = Q[S][A] + W / C[S][A] * (G - Q[S][A])

            AStar = np.random.choice(np.where(Q[S] == np.max(Q[S]))[0])

            for j in range(self.action_shape):
                if j != AStar:
                    pi[S][j] = 0
                else:
                    pi[S][j] = 1

            if A != AStar:
                break
            W = W * 1 / self.b[S][A]


    def save(self, num):
        np.savez("blackjackdata/data_{}_{}.npz".format(self.name, num), Q=np.array(dict(self.Q)), C=np.array(dict(self.C)), pi=np.array(dict(self.pi)))

    def load(self, num):
        file_name = "blackjackdata/data_{}_{}.npz".format(self.name, num)
        if not exists(file_name):
            return False

        npzfile = np.load(file_name, allow_pickle=True)
        self.Q.update(npzfile['Q'].item())
        self.C.update(npzfile['C'].item())
        self.pi.update(npzfile['pi'].item())
        return True

def get_probability():
    a = np.random.random(2)
    return a / np.sum(a)

env1 = gym.make('Blackjack-v1', natural=False, sab=True)
env2 = gym.make('Blackjack-v1', natural=False, sab=True)
env3 = gym.make('Blackjack-v1', natural=False, sab=True)
check_env(env1.unwrapped)
check_env(env2.unwrapped)
check_env(env3.unwrapped)

blackjackOnPolicyAgent = BlackjackOnPolicyAgent(env1, "onpolicy")
blackjackOffPolicyAgent = BlackjackOffPolicyAgent(env2, "offpolicy")
blackjackOffPolicyAgentRandom = BlackjackOffPolicyAgentRandom(env3, "offpolicy-random")

episodes_num = 600000
interval = 100000

def train(agent, episodes_num):
    i = 0
    while i < episodes_num:
        if i % interval == 0:
            if agent.load(i + interval):
                i += interval
                continue

        agent.learn()
        if i and i % interval == 0:
            agent.save(i)

        i += 1

    if i and i % interval == 0:
        agent.save(i)

def show(agent, total_line, total_col, index, name, fig=plt.figure()):
    Q = agent.Q
    pi = agent.pi

    # observation_space = agent.observation_space
    # dimension1, dimension2, _ = observation_space[0].n, observation_space[1].n, observation_space[2].n
    dimension1 = 10
    dimension2 = 10

    # curren sum
    a = np.linspace(12, 22, dimension1, dtype=int, endpoint=False)
    # dealer's one showing card
    b = np.linspace(1, 11, dimension2, dtype=int, endpoint=False)
    x, y = np.meshgrid(b, a)
    # z = np.vectorize(lambda x : Q[x])(np.rec.fromarrays([x, y], names='x,y'))
    z0 = np.zeros((dimension1, dimension2))
    z1 = np.zeros((dimension1, dimension2))

    for i in range(dimension1):
        for j in range(dimension2):
            ti = i + 12
            tj = j + 1
            # action-value -> state-value (we use epsilon-soft policy)
            z0[i, j] = np.sum(Q[(ti, tj, False)] * pi[(ti, tj, False)])
            z1[i, j] = np.sum(Q[(ti, tj, True)] * pi[(ti, tj, True)])

    ax0 = fig.add_subplot(total_line, total_col, index * total_col - 1 , projection='3d')
    ax1 = fig.add_subplot(total_line, total_col, index * total_col, projection='3d')

    ax0.set_title('{} without Usable Ace'.format(name))
    ax1.set_title('{} with Usable Ace'.format(name))

    ax0.set_xlabel("the dealer's one showing card")
    ax0.set_ylabel("player's currer sum")
    ax0.set_zlabel("state-value function")
    ax0.plot_wireframe(x, y, z0)

    ax1.set_xlabel("the dealer's one showing card")
    ax1.set_ylabel("player's currer sum")
    ax1.set_zlabel("state-value function")
    ax1.plot_wireframe(x, y, z1)

train(agent=blackjackOnPolicyAgent, episodes_num=episodes_num)
train(agent=blackjackOffPolicyAgent, episodes_num=episodes_num)
train(agent=blackjackOffPolicyAgentRandom, episodes_num=episodes_num)

total_line, total_col = 3, 2
show(blackjackOnPolicyAgent, total_line, total_col, 1, "On-policy")
show(blackjackOffPolicyAgent, total_line, total_col, 2, "Off-policy equal")
show(blackjackOffPolicyAgentRandom, total_line, total_col, 3, "Off-policy random")
print(blackjackOffPolicyAgent.pi.items())
plt.show()
