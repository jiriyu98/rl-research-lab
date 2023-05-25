from time import sleep
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ref: https://coax.readthedocs.io/en/latest/examples/frozen_lake/ppo.html
# ref: https://github.com/huggingface/deep-rl-class/blob/main/unit5/unit5.ipynb
# ref: http://www.cs.otago.ac.nz/cosc470/09-deep-reinforcement-learning.pdf
# ref: https://groups.google.com/g/rl-list/c/rv92nGy0YRk


def generate_random_map(size=4, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        # res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res = ["SFFF", "FHFH", "FFFH", "HFFF"]
        res = [list(x) for x in res]

        g_x, g_y = np.random.choice(size, 2)
        if g_x == 0 and g_y == 0:
            continue

        res[g_x][g_y] = "G"
        valid = is_valid(res)
    return ["".join(x) for x in res]


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(16, 64)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(64, 128)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(128, 4)),
            ('softMax1', nn.Softmax(dim=-1)),
        ]))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.int64).unsqueeze(0)
        x = x.to(device)
        x = F.one_hot(x, 16)
        x = x.to(torch.float)
        return self.net(x)

    def argforward(self, x, weights):
        x = torch.tensor(x, dtype=torch.int64).unsqueeze(0)
        x = x.to(device)
        x = F.one_hot(x, 16)
        x = x.to(torch.float)
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        x = F.softmax(x, dim=-1)
        return x


class Agent():
    def __init__(self) -> None:
        self.net = PolicyNet()
        self.env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFF"],
                            # map_name="4x4",
                            is_slippery=False,)
        self.beta = 1e-3
        self.gamma = 0.99
        self.every_print = 10
        self.weights = list(self.net.parameters())
        self.opt = torch.optim.Adam(self.weights, self.beta)
        self.path_maml_net = "./param/parametes_MAML"

    def selectAction(self, action_prob):
        m = Categorical(action_prob)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def sample_data(self, weight, size=20):
        rollout = []
        observation = self.env.reset()
        for _ in range(size):
            action_prob = self.net.argforward(observation, weight)
            action, log_prob = self.selectAction(action_prob)
            next_observation, extrinsic_reward, terminated, _ = self.env.step(
                action)

            if next_observation == observation:
                extrinsic_reward = -0.01

            rollout.append({"observation": observation,
                            "action_prob": action_prob,
                            "log_prob": log_prob,
                            "action": action,
                            "extrinsic_reward": extrinsic_reward,
                            "terminated": terminated,
                            })

            observation = next_observation

            if terminated:
                break

        return rollout

    def getLoss(self, rollout):
        discounted_return = 0
        returns = []
        for tuple in rollout[::-1]:
            discounted_return = discounted_return * \
                self.gamma + tuple["extrinsic_reward"]
            returns.append(discounted_return)

        returns = returns[::-1]
        returns = torch.tensor(returns)

        policy_loss = []
        for index in range(len(returns)):
            log_prob, R = rollout[index]["log_prob"], returns[index]
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()

        return discounted_return, policy_loss

    def loadMAML(self):
        self.weights = torch.load(self.path_maml_net)
        self.meta_optimiser = torch.optim.Adam(self.weights, self.beta)

    def train(self):
        print_return = 0
        print_loss = 0
        for episode in range(1, 5000 + 1):
            # rollout = self.sample_data(self.weights)
            # discounted_return, policy_loss = self.getLoss(rollout)

            # self.opt.zero_grad()
            # policy_loss.backward()
            # self.opt.step()

            # Reinforce
            discounted_return = 0
            nondiscouted_return = 0
            policy_loss = 0
            size = 20
            observation = self.env.reset()

            tracer = []

            for _ in range(size):
                action, log_prob = self.selectAction(
                    self.net.argforward(observation, self.weights))
                next_observation, extrinsic_reward, terminated, _ = self.env.step(
                    action)

                nondiscouted_return = nondiscouted_return + extrinsic_reward

                if next_observation == observation:
                    extrinsic_reward = -0.01

                tracer.append((extrinsic_reward, observation, action))

                if terminated:
                    break
                observation = next_observation

            while tracer:
                r, observation, action = tracer.pop()
                discounted_return = self.gamma * discounted_return + r
                log_prob = Categorical(self.net.argforward(observation, self.weights)).log_prob(
                    torch.tensor([action]))

                loss = -log_prob * discounted_return
                policy_loss += loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            # what
            print_return += discounted_return
            print_loss += policy_loss.item()

            if episode % self.every_print == 0:
                print("episode: {}, \t print_return: {:.3f} \t, policy_loss: {:.3f}".format(
                    episode, print_return / self.every_print, print_loss / self.every_print))
                print_return = 0
                print_loss = 0

    def test(self, size=20):
        rollout = []
        observation = self.env.reset()
        for _ in range(size):
            print(self.env.render("ansi"))
            action_prob = self.net.argforward(observation, self.weights)
            [print("{:.3f} : {}".format(x, y)) for (x, y) in zip(
                action_prob[0], ["LEFT", "DOWN", "RIGHT", "UP"])]
            action, _ = self.selectAction(action_prob)
            next_observation, extrinsic_reward, terminated, _ = self.env.step(
                action)
            observation = next_observation

            print("Reward: {} ".format(extrinsic_reward))

            if terminated:
                break

            input()

    def testAllGrids(self):
        for observation in range(16):
            action_prob = self.net.argforward(observation, self.weights)
            [print("{} -> {:.3f} : {}".format(observation, x, y)) for (x, y) in zip(
                action_prob[0], ["LEFT", "DOWN", "RIGHT", "UP"])]


agent = Agent()
agent.env.reset()
agent.env.render("human")

# agent.loadMAML()
# agent.train()
# agent.test()
