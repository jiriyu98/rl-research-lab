from threading import Lock, Thread
from time import sleep

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# cuda
if torch.cuda.is_available():
    torch.cuda.device(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# global config

torch.manual_seed(0)

state_space = 4
action_space = 2
t_max = 10000
T_MAX = t_max * 100
FLAG = True
global_t = 0
local_update = Lock()
running_reward = 10
episode_count = 0
eps = np.finfo(np.float32).eps.item()


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
    # for each thread that will work independently
    def __init__(self, id):
        # will not create actor and critic here, instead, deepcopy actor and critic everytime
        # temparoray
        self.id = id
        self.env = gym.make('CartPole-v1')
        None

    def update(self, local_v: nn.Module, global_v: nn.Module):
        for local_parameter, global_parameter in zip(local_v.parameters(), global_v.parameters()):
            global_parameter._grad = local_parameter.grad

    def trainEpisode(self, global_actor: Actor, global_critic: Critic, actor_optimizer, critic_optimizer, loop_count, gamma=1.0):
        state = self.env.reset()
        rewards = []
        states = [state]
        saved_log_probs = []
        ep_reward = 0
        global global_t, running_reward, episode_count, FLAG

        actor = Actor(state_space, action_space)
        critic = Critic(state_space, action_space)
        actor.load_state_dict(global_actor.state_dict())
        critic.load_state_dict(global_critic.state_dict())

        if (global_t > T_MAX):
            return

        # step1: create transitions
        for _ in range(t_max):
            # step
            action, saved_log_prob = actor(state)
            saved_log_probs.append(saved_log_prob)

            state, reward, done, _ = self.env.step(action.item())

            # save rewards and states
            rewards.append(reward)
            states.append(state)
            ep_reward += reward

            # exit when it's done
            if done:
                break

        # step2: generate loss
        R = 0
        states.pop()
        rewards = rewards[::-1]
        returns = []  # when it is terminal, retusn should be zero
        actor_loss = []
        critic_loss = []

        for r in rewards:
            R = r + gamma * R
            returns.append(R)
        returns = returns[::-1]
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # advantage: R - state-value-func(state)
        # w = w + 2 * advantage * d theta (advantage)
        for i in range(len(returns)):
            advantage = R - critic(states[i])
            actor_loss.append(-saved_log_probs[i] * advantage.detach())
            critic_loss.append(2 * advantage * advantage.detach())
            # TODO: check it later
            # critic_loss.append(advantage * advantage)

        # step3: update global vars by local vars
        # atomic
        local_update.acquire()  # lock

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        actor_loss = torch.cat(actor_loss).sum()
        actor_loss.to(device)
        actor_loss.backward()

        critic_loss = torch.cat(critic_loss).sum()
        critic_loss.to(device)
        critic_loss.backward()

        self.update(actor, global_actor)
        self.update(critic, global_critic)

        actor_optimizer.step()
        critic_optimizer.step()

        # update global_t, running_reward
        global_t += len(rewards)
        episode_count += 1
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # stop
        if running_reward > self.env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, len(states)))
            FLAG = False

        # log
        if loop_count % 10 == 0:
            print("now thread {:2d}  is updating, episode_count: {} \t running reward: {:.2f}"
                  .format(self.id, episode_count, running_reward))

        # if loop_count % 10 == 0:
        #     for parameter in global_actor.parameters():
        #         print(parameter)
        #     sleep(1)

        local_update.release()  # unlock

        return

    def trainLoop(self, global_actor, global_critic, actor_optimizer, critic_optimizer, gamma=1.0):
        loop_count = 0

        while global_t <= T_MAX and FLAG:
            loop_count += 1
            self.trainEpisode(global_actor, global_critic, actor_optimizer,
                              critic_optimizer, loop_count)


class Coordinator:
    # Node for coordinating
    # in A3C, it does nothing.

    def __init__(self) -> None:
        self.actor = Actor(state_space, action_space)
        self.critic = Critic(state_space, action_space)
        # this config probably needs 700 epsidoes to converge
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-3)
        # make advantage to be trivial
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-2)

    def train(self):
        worker_num = 3
        workers = []
        worker_threads = []

        for id in range(worker_num):
            worker = Worker(id)
            workers.append(worker)

            t = Thread(target=worker.trainLoop, args=(
                self.actor, self.critic, self.actor_optimizer, self.critic_optimizer))
            t.start()
            worker_threads.append(t)

        for thread in worker_threads:
            thread.join()


def main():
    coordinator = Coordinator()

    coordinator.train()


if __name__ == '__main__':
    main()
