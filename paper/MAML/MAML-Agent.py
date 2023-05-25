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
device = torch.device("cpu")

# ref: https://github.com/GauravIyer/MAML-Pytorch/blob/master/Experiment%201/Experiment_1_Sine_Regression.ipynb
# ref: https://coax.readthedocs.io/en/latest/examples/frozen_lake/ppo.html


def generate_random_map(size=8, p=0.8):
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
        if (g_x == 0 and g_y == 0) or res[g_x][g_y] == 'H':
            continue

        res[g_x][g_y] = "G"
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeSingleTask():
    def __init__(self, net, desc) -> None:
        self.env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False)
        self.net = net

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


class FrozenLakeDistribution():
    def __init__(self) -> None:
        pass

    def sample_task(self, net):
        return FrozenLakeSingleTask(net, generate_random_map(size=4))


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
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        return self.net(x)

    # I implemented argforward() so that I could use a set of custom weights for evaluation.
    # This is important for the "inner loop" in MAML where you temporarily update the weights
    # of the network for a task to calculate the meta-loss and then reset them for the next meta-task.

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


class CusMAML():
    def __init__(self, net, alpha, beta, tasks, k, num_metatasks):
        self.net = net
        self.weights = list(net.parameters())
        self.init_weights = list(net.parameters())
        self.alpha = alpha
        self.beta = beta
        self.tasks = tasks
        self.k = k
        self.num_tasks_meta = num_metatasks
        self.meta_optimiser = torch.optim.Adam(self.weights, self.beta)
        self.print_every = 100
        self.num_metatasks = num_metatasks
        self.gamma = 0.99
        self.path_maml_net = "./param/parametes_MAML2"

    def saveMAML(self):
        torch.save(self.weights, self.path_maml_net)

    def loadMAML(self):
        self.weights = torch.load(self.path_maml_net)
        self.meta_optimiser = torch.optim.Adam(self.weights, self.beta)

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

    def inner_loop(self, task, count_map=None):
        metaloss = 0
        discounted_return_sum = 0
        def to_index(x): return (x // 4, x % 4)

        for _ in range(self.k):
            temp_weights = [w.clone() for w in self.weights]
            rollout = task.sample_data(
                weight=temp_weights)  # sampling D

            if count_map is not None:
                for transition in rollout:
                    count_map[to_index(transition["observation"])] += 1

            discounted_return, loss = self.getLoss(rollout)
            discounted_return_sum += discounted_return / self.k

            loss = loss.to(device)
            grads = torch.autograd.grad(loss, temp_weights)
            # temporary update of weights
            temp_weights = [w-self.alpha*g for w,
                            g in zip(temp_weights, grads)]
            rollout = task.sample_data(weight=temp_weights)  # sampling D'
            _, metaloss_tmp = self.getLoss(rollout)
            metaloss += metaloss_tmp

        return discounted_return_sum, metaloss

    def outer_loop(self, num_epochs, count_map=None):
        total_loss = 0
        average_return = 0

        # line 2
        for epoch in range(1, num_epochs+1):
            metaloss_sum = 0
            discounted_return_sum = 0

            # line 4 - line 9
            for _ in range(self.num_metatasks):
                task = self.tasks.sample_task(self.net)
                discounted_return_tmp, metaloss = self.inner_loop(
                    task, count_map)
                metaloss_sum += metaloss
                discounted_return_sum += discounted_return_tmp
            metaloss_sum = metaloss_sum.to(device)
            metagrads = torch.autograd.grad(metaloss_sum, self.weights)

            # line 10
            for w, g in zip(self.weights, metagrads):
                w.grad = g

            ###############
            self.meta_optimiser.step()
            total_loss += metaloss_sum.item() / self.num_metatasks
            average_return += discounted_return_sum / self.num_metatasks
            if epoch % self.print_every == 0:
                print("{}/{}. loss: {:.3f}, average_return: {:.3f}".format(epoch,
                      num_epochs, total_loss / self.print_every, average_return / self.print_every))
                total_loss = 0
                average_return = 0
                self.saveMAML()
                if count_map is not None:
                    self.heatmap(
                        count_map, name="after {} gradient updates.png".format(epoch), cmap="YlGn", cbarlabel="visitcount")

    def adaptToNewTask(self, debug=False, desc=None, adaption_num=20):
        # task = self.tasks.sample_task(self.net)
        if desc is None:
            task = self.tasks.sample_task(self.net)
        else:
            task = FrozenLakeSingleTask(self.net, desc)

        # task.env.reset()
        # print(task.env.render("ansi"))

        # prev
        average_return_prev = 0
        for _ in range(self.print_every):
            rollout = task.sample_data(self.weights)
            discounted_return, _ = self.getLoss(rollout)
            average_return_prev += discounted_return
        if debug:
            print("before training - discounted_return: {:.3f}".format(
                average_return_prev / self.print_every))

        # adaption
        for _ in range(adaption_num):
            loss = 0
            rollout = task.sample_data(self.weights)
            _, policy_loss = self.getLoss(rollout)
            loss += policy_loss
            self.meta_optimiser.zero_grad()
            loss.backward()
            self.meta_optimiser.step()

        # post
        average_return_post = 0
        for _ in range(self.print_every):
            rollout = task.sample_data(self.weights)
            discounted_return, _ = self.getLoss(rollout)
            average_return_post += discounted_return
        if debug:
            print("after training - discounted_return: {:.3f}".format(
                average_return_post / self.print_every))

        return average_return_prev / self.print_every, average_return_post / self.print_every

    def adaptToNewTask_vanilla(self, debug=False, desc=None, adaption_num=20):
        # new weights
        self.weights = []
        for w in self.init_weights:
            w = w.clone().detach()
            w.requires_grad = True
            self.weights.append(w)
        self.meta_optimiser = torch.optim.Adam(self.weights, self.beta)

        # task = self.tasks.sample_task(self.net)
        if desc is None:
            task = self.tasks.sample_task(self.net)
        else:
            task = FrozenLakeSingleTask(self.net, desc)

        # prev
        average_return_prev = 0
        for _ in range(self.print_every):
            rollout = task.sample_data(self.weights)
            discounted_return, _ = self.getLoss(rollout)
            average_return_prev += discounted_return
        if debug:
            print("before training - discounted_return: {:.3f}".format(
                average_return_prev / self.print_every))

        # adaption
        for _ in range(adaption_num):
            loss = 0
            rollout = task.sample_data(self.weights)
            _, policy_loss = self.getLoss(rollout)
            loss += policy_loss
            self.meta_optimiser.zero_grad()
            loss.backward()
            self.meta_optimiser.step()

        # post
        average_return_post = 0
        for _ in range(self.print_every):
            rollout = task.sample_data(self.weights)
            discounted_return, _ = self.getLoss(rollout)
            average_return_post += discounted_return
        if debug:
            print("after training - discounted_return: {:.3f}".format(
                average_return_post / self.print_every))

        return average_return_prev / self.print_every, average_return_post / self.print_every

    def adaptToNewTasks(self, num_map):
        average_returns = []
        for _ in range(1, num_map+1):
            average_return_prev, average_return_post = self.adaptToNewTask(
                True)
            average_returns.append((average_return_prev, average_return_post))

        print(average_returns)
        for x in zip(*average_returns):
            plt.scatter(np.arange(num_map), x)

        plt.show()

    def adaptToEveryTask(self):
        descs = [
            ["SFFF", "FHFH", "FFFH", "HFFG"],
            ["SFFF", "FHFH", "FFFH", "HFGF"],
            ["SFFF", "FHFH", "FFFH", "HGFF"],
            ["SFFF", "FHFH", "FFGH", "HFFF"],
            ["SFFF", "FHFH", "FGFH", "HFFF"],
            ["SFFF", "FHFH", "GFFH", "HFFF"],
            ["SFFF", "FHGH", "FFFH", "HFFF"],
            ["SFFF", "GHFH", "FFFH", "HFFF"],
            ["SFFG", "FHFH", "FFFH", "HFFF"],
            ["SFGF", "FHFH", "FFFH", "HFFF"],
            ["SGFF", "FHFH", "FFFH", "HFFF"],
        ]

        average_returns = []
        adaption_num = 5
        labels = ["prev_adaption_maml", "post_adaption_maml",
                  "prev_adaption_vanilla", "post_adaption_vanilla"]
        for desc in descs:
            self.loadMAML()
            average_return_prev, average_return_post = self.adaptToNewTask(
                False, desc, adaption_num)

            average_return_prev_vanilla, average_return_post_vanilla = self.adaptToNewTask_vanilla(
                False, desc, adaption_num)

            average_returns.append((average_return_prev, average_return_post,
                                   average_return_prev_vanilla, average_return_post_vanilla))

        for x, label in zip(zip(*average_returns), labels):
            plt.scatter(np.arange(len(descs)), x, label=label)

        plt.title("adaption with {} trajectories".format(adaption_num))
        plt.legend()
        plt.show()

    # ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    def heatmap(self, data, name="temp", row_labels=None, col_labels=None, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.savefig("./heatmap/" + name)
        plt.clf()

# train


tasks = FrozenLakeDistribution()
net = PolicyNet()
net = net.to(device)
maml = CusMAML(net, alpha=0.01, beta=0.001,
               tasks=tasks, k=5, num_metatasks=10)
count_map = np.zeros((4, 4,))
maml.outer_loop(num_epochs=5000, count_map=count_map)
maml.saveMAML()

# adaption

# net = PolicyNet()
# tasks = FrozenLakeDistribution()
# maml = CusMAML(net, alpha=0.01, beta=0.001,
#                tasks=tasks, k=5, num_metatasks=10)

# maml.heatmap(count_map, cmap="YlGn", cbarlabel="visitcount")
# maml.adaptToEveryTask()
