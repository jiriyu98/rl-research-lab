from time import sleep
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors

from torch.distributions import Categorical
# from gym.envs.toy_text.frozen_lake import generate_random_map


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# ref: https://github.com/GauravIyer/MAML-Pytorch/blob/master/Experiment%201/Experiment_1_Sine_Regression.ipynb
# ref: https://coax.readthedocs.io/en/latest/examples/frozen_lake/ppo.html
# ref-meta-update: https://discuss.pytorch.org/t/how-to-manually-update-network-parameters-while-keeping-track-of-its-computational-graph/131642

# map


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

    def random_choice_unrepeatable(matrix, num):
        rows, cols = size, size
        avaliable_positions = [(row, col) for row in range(rows)
                               for col in range(cols) if matrix[row][col] == 'F']
        return [avaliable_positions[x] for x in np.random.choice(len(avaliable_positions), 2, replace=False)]

    while not valid:
        p = min(1, p)
        # res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        # res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res = ["FFFF", "FHFH", "FFFH", "HFFF"]
        res = [list(x) for x in res]

        (s_x, s_y), (g_x, g_y) = random_choice_unrepeatable(res, 2)

        res[s_x][s_y] = "S"
        res[g_x][g_y] = "G"
        valid = is_valid(res)

    return ["".join(x) for x in res]


class FrozenLakeSingleTask():
    def __init__(self, desc) -> None:
        self.env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False)
        # inner discount factor (intrinsic reward)
        self.inner_task_discount = 0.9
        # outer discount factor (extrinsic reward)
        self.outer_task_discount = 0.99

    def render(self):
        self.env.reset()
        print(self.env.render("ansi"))

    def selectAction(self, action_prob):
        m = Categorical(action_prob)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def generateEpisode(self, policy, size=20):
        episode = []
        observation = self.env.reset()
        for _ in range(size):
            action_prob, _ = policy(observation)
            action, _ = self.selectAction(action_prob)
            next_observation, extrinsic_reward, terminated, _ = self.env.step(
                action)

            if next_observation == observation:
                extrinsic_reward = -0.01

            episode.append({"observation": observation,
                            "action": action,
                            "extrinsic_reward": extrinsic_reward,
                            "next_observation": next_observation,
                            "terminated": terminated,
                            })

            observation = next_observation

            if terminated:
                break

        return episode

    def getRollouts(self, size, length_trajectory, N, policy, episode_limit):
        # size: episode length limit
        # weight: policy weight
        # length_trajectory: length of rollout (trajectory)
        # N: number of rollouts
        rollouts = []

        episode_count = 0

        while len(rollouts) < N and episode_count < episode_limit:
            episode = self.generateEpisode(policy, size)
            episode_count += 1
            length_episode = len(episode)
            reminder = length_episode % length_trajectory
            for i in range(0, length_episode - reminder, length_trajectory):
                rollouts.append(episode[i:i+length_trajectory])
            if reminder:
                rollouts.append(episode[-reminder:])

        return rollouts[:N], episode_count


class FrozenLakeDistribution():
    def __init__(self) -> None:
        pass

    def sample_task(self) -> FrozenLakeSingleTask:
        return FrozenLakeSingleTask(generate_random_map(size=4))


# model

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(16, 64)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(64, 128)),
            ('relu2', nn.ReLU()),
        ]))

        self.action_head = nn.Sequential(OrderedDict([
            ('action_l1', nn.Linear(128, 4)),
            ('softMax1', nn.Softmax(dim=-1)),
        ]))
        self.value_head = nn.Sequential(OrderedDict([
            ('value_l1', nn.Linear(128, 1)),
        ]))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.int64).unsqueeze(0)
        x = x.to(device)
        x = F.one_hot(x, 16)
        x = x.to(torch.float)
        x = self.net(x)

        return self.action_head(x), self.value_head(x)


class IntrinsicRewardAndLifetimeValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_hidden_size = 2
        self.observation_dim = 16
        self.action_dim = 4

        self.net = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(16, 128)),
            ('relu1', nn.ReLU()),
            # ('l2', nn.Linear(64, 128)),
            # ('relu2', nn.ReLU()),
            ('lstm', nn.LSTM(128, self.rnn_hidden_size)),
        ]))

    def forward(self, s0, hidden_state=(None, None)):
        # input adjust
        s0 = torch.tensor(s0).unsqueeze(0).detach()
        s0 = F.one_hot(s0, self.observation_dim)
        s0 = s0.to(torch.float)

        x = s0

        # split
        fc, lstm = self.net[:-1], self.net[-1]

        x = fc(x)
        x, (hn, cn) = lstm(x, hidden_state)
        intrinsic_reward, lifetime_value = x.squeeze(0)

        # intrinsicreward and history info
        return intrinsic_reward, lifetime_value, (hn.detach(), cn.detach())

# agent


class Agent():
    def __init__(self) -> None:
        self.env_dist = FrozenLakeDistribution()
        self.lifetime_episode_limit = 200
        self.lifetime_episode_left = 200
        self.episode_length_limit = 20
        self.trajectory_length = 8  # trajectory length (T)
        self.rollouts_number = 5  # outer unroll length (N)

        self.lifetime_mask_count = 0

        self.lifetime_train_batch = 40

        # IntrinsicRewardAndLifetimeValue
        self.intrinsic_reward_and_lifetime_value = IntrinsicRewardAndLifetimeValue()
        self.hidden_state = None
        self.alpha = 0.01
        self.beta = 0.001
        self.opt = torch.optim.Adam(
            self.intrinsic_reward_and_lifetime_value.parameters(), lr=self.beta, eps=1e-3)

        self.print_every = 200
        self.total_episode_count = 0
        self.count_map = np.zeros((4, 4,))

    def save_intrinsic_reward(self):
        pass

    def load_intrinsic_reward(self):
        pass

    def policy_gradient_loss_fn(self, trajectory, values, logits, discounted_returns):
        trajectory_length = len(trajectory)

        # get discounted loss and adv
        advs = []
        for i in range(trajectory_length):
            advs.append(discounted_returns[i] - values[i])

        policy_loss = 0
        for i in range(trajectory_length):
            trans = trajectory[i]
            action = trans["action"]

            # logits -> log_prob
            log_prob = Categorical(logits[i]).log_prob(torch.tensor(action))
            policy_loss += -log_prob * advs[i]

        return policy_loss

    def get_rollouts(self, task: FrozenLakeSingleTask, policy: PolicyNet):
        rollouts, episode_count = task.getRollouts(
            self.episode_length_limit, self.trajectory_length, self.rollouts_number, policy, self.lifetime_episode_left)
        self.lifetime_episode_left -= episode_count
        self.total_episode_count += episode_count

        return rollouts

    def inner_loop(self, task: FrozenLakeSingleTask, policy: PolicyNet):
        rollouts = self.get_rollouts(task, policy)

        # rollouts can be in many shapes so there is no check/assert

        for rollout in rollouts:
            rollout_length = len(rollout)

            # rnn
            intrinsic_rewards = []
            for j in range(rollout_length):
                trans = rollout[j]
                obs = trans["observation"]

                intrinsic_reward, _, self.hidden_state = self.intrinsic_reward_and_lifetime_value(
                    obs, self.hidden_state)
                intrinsic_rewards.append(intrinsic_reward)

            # logits and state_values
            logits = []
            values = []
            for j in range(rollout_length):
                trans = rollout[j]
                obs = trans["observation"]
                logit, state_value = policy(obs)
                logits.append(logit)
                values.append(state_value)

                if j == rollout_length - 1:
                    last_trans = rollout[-1]

                    if last_trans["terminated"]:
                        values.append(0)
                    else:
                        _, state_value = policy(last_trans["next_observation"])
                        values.append(state_value)

            assert (len(values) == rollout_length + 1)

            # discounted_returns
            discounted_returns = []
            discounted_acc = 0
            bootstrap = values[-1]
            for j in range(rollout_length-1, -1, -1):
                trans = rollout[j]
                obs = trans["observation"]
                intrinsic_reward = intrinsic_rewards[j]
                discounted_acc = discounted_acc * task.inner_task_discount + intrinsic_reward
                bootstrap = bootstrap * task.inner_task_discount
                discounted_returns.append(discounted_acc + bootstrap)
            discounted_returns = discounted_returns[::-1]

            policy_loss = self.policy_gradient_loss_fn(
                rollout, values, logits, discounted_returns)

            baseline_loss = 0
            for j in range(rollout_length):
                baseline_loss += 0.5 * \
                    torch.square(discounted_returns[j] - values[j])

            loss = policy_loss + baseline_loss

            # update policy
            # ref: https://github.com/learnables/learn2learn/blob/06893e847693a0227d5f35a6e065e6161bb08201/learn2learn/utils/__init__.py
            policy_parameter_grads = torch.autograd.grad(
                loss, policy.parameters(), create_graph=True)
            for grad in policy_parameter_grads:
                torch.clamp(grad, -0.1, 0.1)

            updates = [w - self.alpha * g for w,
                       g in zip(policy.parameters(), policy_parameter_grads)]

            self.update_gradient(policy, updates)

        return rollouts

    def update_gradient(self, module, updates=None):
        if updates is not None:
            params = list(module.parameters())
            for p, g in zip(params, updates):
                p.update = g

        for param_key in module._parameters:
            p = module._parameters[param_key]
            if p is not None and hasattr(p, 'update') and p.update is not None:
                module._parameters[param_key] = p.update

        for module_key in module._modules:
            module._modules[module_key] = self.update_gradient(
                module._modules[module_key],
                updates=None,
            )

        return module

    def fix_parameters(self, module):
        for param_key in module._parameters:
            p = module._parameters[param_key].clone().detach()
            p.requires_grad = True
            module._parameters[param_key] = p

        for module_key in module._modules:
            module._modules[module_key] = self.fix_parameters(
                module._modules[module_key],
            )

        return module

    def outer_loop(self, policy, task: FrozenLakeSingleTask):
        # outer loop
        self.reset()
        while self.lifetime_episode_left > 0:
            rollouts = self.inner_loop(task, policy)

            # flatten: update intrinsic reward and lifetime value
            rollout = []
            for rl in rollouts:
                rollout += rl
            rollout_length = len(rollout)

            # rnn -> values
            values = []
            for j in range(rollout_length):
                trans = rollout[j]
                obs = trans["observation"]

                _, lifetime_value, _ = self.intrinsic_reward_and_lifetime_value(
                    obs, self.hidden_state)
                values.append(lifetime_value)

            # the last trans is only for bootstrapping
            rollout_length -= 1
            rollout.pop()

            # logits
            logits = []
            for j in range(rollout_length):
                trans = rollout[j]
                obs = trans["observation"]
                logit, _ = policy(obs)
                logits.append(logit)

            # discounted_returns
            discounted_returns = []
            discounted_acc = 0
            lifetime_mask = task.outer_task_discount ** self.lifetime_mask_count
            bootstrap = values[-1]
            for j in range(rollout_length-1, -1, -1):
                trans = rollout[j]
                obs = trans["observation"]
                extrinsic_reward = trans["extrinsic_reward"]
                discounted_acc = discounted_acc * task.outer_task_discount + extrinsic_reward
                bootstrap = bootstrap * task.outer_task_discount
                discounted_returns.append(discounted_acc + bootstrap)
            discounted_returns = discounted_returns[::-1]

            # maintain the mask count
            self.lifetime_mask_count += rollout_length

            policy_loss = self.policy_gradient_loss_fn(
                rollout, values, logits, discounted_returns)

            baseline_loss = 0
            for j in range(rollout_length):
                baseline_loss += 0.5 * \
                    torch.square(discounted_returns[j] - values[j])

            # gradient update
            loss = policy_loss + baseline_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # fianlly erase all the gradient info in the policy to avoid backwarding twice
            self.fix_parameters(policy)

            # log
            for transition in rollout:
                self.count_map[(lambda x: (x // 4, x % 4))
                               (transition["observation"])] += 1
                if transition["terminated"] and transition["extrinsic_reward"] > 0:
                    self.count_map[(lambda x: (x // 4, x % 4))(
                        transition["next_observation"])] += 1

        if self.total_episode_count % self.lifetime_episode_limit == 0:
            self.heatmap(self.count_map, task, name="after {} episodes.png".format(
                self.total_episode_count), cmap="YlGn", cbarlabel="visitcount")
            self.count_map = np.zeros((4, 4,))
            print("{}/{}. loss: {:.3f}".format(self.total_episode_count,
                                               self.lifetime_episode_limit * self.lifetime_train_batch, loss.item()))

    def reset(self):
        self.lifetime_episode_left = self.lifetime_episode_limit
        self.hidden_state = None

    def test(self):
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
        self.total_episode_count = 0
        for desc in descs:
            # init a policy
            policy = PolicyNet()
            # init a task
            task = FrozenLakeSingleTask(desc)
            self.get_average_return(policy, task, "before training", 30)
            self.outer_loop(policy, task)
            self.get_average_return(policy, task, "after training", 30)
            print()

    def get_average_return(self, policy: PolicyNet, task: FrozenLakeSingleTask, output_str, total_count=None):
        if total_count is None:
            total_count = self.lifetime_episode_limit

        acc_return = 0
        for _ in range(total_count):
            episode = task.generateEpisode(policy)
            discounted_return = sum(x["extrinsic_reward"] for x in episode)
            acc_return += discounted_return
        print("{} - average discounted return: {:.3f}".format(output_str,
                                                              acc_return / total_count))

    def train(self):
        self.total_episode_count = 0
        for _ in range(1, self.lifetime_train_batch+1):
            # init a policy
            policy = PolicyNet()
            # init a task
            task = self.env_dist.sample_task()
            self.outer_loop(policy, task)

    def heatmap(self, data, task, name="temp", row_labels=None, col_labels=None, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
        if ax is None:
            # ax = plt.gca()
            _, (_, ax) = plt.subplots(1, 2)

        if cbar_kw is None:
            cbar_kw = {}

        # task

        desc = task.env.desc.copy().tolist()
        desc_dict = {b'S': 0, b'F': 1, b'H': 2, b'G': 3}
        m, n = len(desc), len(desc[0])
        for i in range(m):
            for j in range(n):
                desc[i][j] = desc_dict[desc[i][j]]
        desc = np.array(desc)
        cmap = colors.ListedColormap(['green', 'white', 'black', 'blue'])
        ax_task = plt.subplot(121)
        # ax_task.get_xaxis().set_visible(False)
        # ax_task.get_yaxis().set_visible(False)
        im = ax_task.imshow(desc, cmap="PuBuGn")

        # Show all ticks and label them with the respective list entries.
        ax_task.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax_task.set_yticks(np.arange(data.shape[0]), labels=row_labels)

        # Let the horizontal axes labeling appear on top.
        ax_task.tick_params(top=True, bottom=False,
                            labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax_task.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax_task.spines[:].set_visible(False)

        ax_task.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax_task.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax_task.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax_task.tick_params(which="minor", bottom=False, left=False)

        # heatmap

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

        # save as png
        plt.savefig("./heatmap/" + name)
        plt.clf()


agent = Agent()
agent.test()
