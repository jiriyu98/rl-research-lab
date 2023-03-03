from threading import Lock, Thread
from time import sleep

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# global config
# state_space = 250  # (250, 160, 3)
# action_space = 18


# class Actor(nn.Module):
#     def __init__(self, state_space, action_space):
#         super().__init__()
#         self.conv1 = nn.Conv2d(state_space, 32, 8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
#         self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
#         self.linear1 = nn.Linear(32 * 7 * 7, 512)
#         self.linear2 = nn.Linear(512, action_space)

#     def forward(self, x):
#         x = torch.from_numpy(x).float().unsqueeze(0)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = flatten

#         x = F.softmax(x, dim=1)
#         m = Categorical(x)

#         action = m.sample()

#         return action, m.log_prob(action)

state_space = 4
action_space = 2


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.affine1 = nn.Linear(state_space, 64)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(64, action_space)

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = F.softmax(x, dim=1)
        m = Categorical(x)

        action = m.sample()
        # save the log_prob

        return action, m.log_prob(action)


class Critic(nn.Module):
    # estimate V state-value function instead of Q action-value function
    def __init__(self, state_space, action_space):
        super().__init__()
        self.affine1 = nn.Linear(state_space, 64)
        self.affine2 = nn.Linear(64, 128)
        self.affine3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.affine3(x)
        return x


class Worker:
    def __init__(self, render=False) -> None:
        if render:
            self.env = gym.make("CartPole-v1", render_mode="human")
        else:
            self.env = gym.make("CartPole-v1")

        # ref: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/train.py
        # ref: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
        self.T = 10
        self.actor = Actor(state_space, action_space)
        self.critic = Critic(state_space, action_space)
        self.batchsize = 50
        self.epoch_num = 80
        self.eps_clip = 0.2
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_training_time = 100000

    def train(self, gamma=1.0):
        state, _ = self.env.reset()
        done = False
        episode_num = 0
        episode_reward = 0

        while not done:
            # a new epsidoe starts
            episode_num += 1

            transitions = []

            # step1: generate transitions
            for i in range(self.T):
                action, saved_log_prob = self.actor(state)
                next_state, reward, done, _, _ = self.env.step(action.item())
                transitions.append(
                    (state, action, reward, saved_log_prob))

                episode_reward += reward

                if done:
                    break

                state = next_state

                # if self.render:
                #     self.env.render()

            for _ in range(self.epoch_num):
                # step2: get advantage (brute-force)
                advantages = [0]
                advantage = 0
                for i in range(len(transitions) - 2, -1, -1):
                    delta = reward + gamma * \
                        self.critic(transitions[i + 1][0]) - \
                        self.critic(transitions[i][0])
                    advantage = advantage * gamma + delta
                    advantages.append(advantage)

                # step3: update actor and critic
                # (trivial adam, no minibatch)
                loss_clip = 0
                for i in range(len(transitions)):
                    log_theta_old = transitions[i][3]
                    _, log_theta = self.actor(state)
                    ratio = torch.exp(log_theta - log_theta_old.detach())

                    # surr
                    surr1 = ratio * advantages[i]
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                                        1 + self.eps_clip) * advantages[i]

                    # loss clip
                    loss_clip = loss_clip - torch.min(surr1, surr2)

                # update
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                loss_clip /= len(transitions)  # get mean
                loss_clip.backward()

                self.optimizer_actor.step()
                self.optimizer_critic.step()

        return episode_reward

    def trainLoop(self):
        running_reward = 0
        for i in range(self.max_training_time):
            # for _ in range(self.epoch_num):
            #     episode_reward = self.train()
            running_reward = running_reward * 0.95 + self.train() * 0.05

            if i % 10 == 0:
                print("training time: {} \t running reward: {:.2f}".format(
                    i, running_reward))


def main():
    worker = Worker()
    worker.trainLoop()


if __name__ == '__main__':
    main()
