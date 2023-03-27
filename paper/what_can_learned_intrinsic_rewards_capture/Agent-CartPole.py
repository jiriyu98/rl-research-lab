from itertools import count

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from envs.CartPole import CartPole
from matplotlib import pyplot as plt
from models.IntrinsicRewardAndLifetimeValue import \
    IntrinsicRewardAndLifetimeValue
from models.Policy import Policy
from torch.distributions import Categorical

'''
Actor-Critic style
'''
torch.autograd.set_detect_anomaly(True)


class Agent():
    def __init__(self, env: CartPole):
        # enviornment
        self.env = env

        # model
        self.state_space = 4
        self.action_space = 2
        self.input_space = 7
        self.policy = Policy(state_space=self.state_space,
                             action_space=self.action_space)
        self.intrinsic_reward_and_lifeimte_value = IntrinsicRewardAndLifetimeValue(
            input_space=self.input_space)

        # hyperparameters
        self.episode_step_limit = 1000
        self.episode_num_per_lifetime = 100
        self.lifetime_done = False
        self.trajectory_length = 8  # n-step trajectory for intrinsic reward udpate
        self.outer_gamma = 0.99
        self.inner_gamma = 0.9
        self.outer_unroll_length = 5
        self.intrinsic_reward_and_lifeimte_value_h0 = None
        self.seed = None

        # utils
        self.adam_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_step_count = 0

        self.adam_m = {}
        self.adam_v = {}

        for module_name, module in self.policy.named_children():
            self.adam_m[module_name] = {}
            self.adam_v[module_name] = {}
            for name_param, param in module.named_parameters():
                self.adam_m[module_name][name_param] = torch.zeros_like(param)
                self.adam_v[module_name][name_param] = torch.zeros_like(param)

        # saved lifetime
        self.lifetime_trajectory = []

        # optimizer for different parameters
        self.policy_optimizer = optim.Adam(self.policy.parameters())
        self.intrinsic_reward_and_lifeimte_value_optimizer = optim.Adam(
            self.intrinsic_reward_and_lifeimte_value.parameters())

        # env states
        self.env_observation = None
        self.env_info = None

    def selectAction(self, action_prob):
        m = Categorical(action_prob)
        action = m.sample()
        return action.item()

    def generateATrajectory(self):
        # initialize the start point
        observation, info = self.env_observation, self.env_info
        trajectory = []

        for index in range(self.trajectory_length + 1):
            action_prob, _ = self.policy(observation["agent"])
            action = self.selectAction(action_prob)
            next_observation, extrinsic_reward, terminated, _, info = self.env.step(
                action)
            self.episode_rewards.append(extrinsic_reward)
            self.episode_step_count += 1

            if self.episode_step_count >= self.episode_step_limit:
                terminated = True

            trajectory.append({"observation": observation["agent"],
                               "action": action,
                               "extrinsic_reward": extrinsic_reward,
                               "terminated": terminated,
                               "debug_index": index,
                               })

            # prepare for next iteration
            observation = next_observation

            if terminated:
                self.env_observation, self.env_info = self.env.reset(
                    seed=self.seed)
                self.episode_count += 1

                # get episode return
                episode_return = 0
                for episode_reward in self.episode_rewards[::-1]:
                    episode_return = self.outer_gamma * episode_return + episode_reward

                self.episode_returns.append(episode_return)
                self.episode_rewards.clear()
                self.episode_step_count = 0

                break

        if self.episode_count >= self.episode_num_per_lifetime:
            self.lifetime_done = True

        self.env_observation = observation
        self.env_info = info

        return trajectory

    def discountedReturnFn(self, trajectory, gamma, bootstrap_value, reward_key):
        # Args:
        # trajectory: T + 1 length
        # return_key: can be "extrinsic_reward" or "intrinsic_reward"

        returns = []
        trajectory_return = bootstrap_value

        for index in range(len(trajectory)):
            trajectory_return = trajectory_return * \
                gamma + trajectory[index][reward_key]
            returns.append(trajectory_return)

        return returns[::-1]

    def feedToRNN(self, trajectory):
        # Args:
        # trajectory: T + 1 length

        # update RNN and state_value, prepare for policy udpate
        for tuple in trajectory:
            observation = tuple["observation"]
            extrinsic_reward = tuple["extrinsic_reward"]
            action = tuple["action"]
            terminated = tuple["terminated"]

            # get intrinsic_reward and lifetime_value
            intrinsic_reward, lifetime_value, self.intrinsic_reward_and_lifeimte_value_h0 = self.intrinsic_reward_and_lifeimte_value(
                (observation, action, extrinsic_reward, terminated), self.intrinsic_reward_and_lifeimte_value_h0)

            _, state_value = self.policy(observation)
            # add intrinsic_reward and lifetime_value into tuple
            tuple["intrinsic_reward"] = intrinsic_reward
            tuple["lifetime_value"] = lifetime_value
            tuple["state_value"] = state_value

    def trainTrajectory(self, trajectory):
        # Args:
        # trajectory: T + 1 length

        # update policy
        bootstrap_value = trajectory[-1]["state_value"] if not trajectory[-1]["terminated"] else torch.zeros_like(
            trajectory[-1]["state_value"])
        returns = self.discountedReturnFn(trajectory=trajectory[-2::-1],  # drop the last one
                                          gamma=self.inner_gamma,
                                          bootstrap_value=bootstrap_value,
                                          reward_key="extrinsic_reward")

        policy_and_value_objective = 0
        for index in range(len(trajectory) - 1):
            action_prob, state_value = self.policy(
                trajectory[index]["observation"])
            log_prob = Categorical(action_prob).log_prob(
                torch.tensor(trajectory[index]["action"]))

            advantage = (returns[index] - state_value)

            policy_loss = (self.inner_gamma ** index) * \
                advantage * log_prob
            value_loss = F.smooth_l1_loss(
                returns[index], state_value)
            policy_and_value_objective += -policy_loss + value_loss

        # grad will not be in the variable now
        policy_parameter_grads = torch.autograd.grad(
            policy_and_value_objective, self.policy.parameters(), create_graph=True)

        grads = {}
        for (name, _), grad in zip(self.policy.named_parameters(), policy_parameter_grads):
            grads[name] = grad

        for module_name, module in self.policy.named_children():
            for param_key in module._parameters:
                param = module._parameters[param_key]
                grad = grads[module_name + '.' + param_key]

                self.adam_count += 1
                correct1 = 1 - .9 ** self.adam_count
                correct2 = 1 - .999 ** self.adam_count

                lr = 1e-5
                m, v = self.adam_m[module_name][param_key], self.adam_v[module_name][param_key]
                m = .9 * m + grad * (1 - .9)
                v = .999 * v + torch.square(grad) * (1 - .999)
                module._parameters[param_key] = param - \
                    m / correct1 * lr / \
                    (torch.sqrt(v / correct2 + 1e-9) + 1e-5)
                self.adam_m[module_name][param_key], self.adam_v[module_name][param_key] = m.detach(
                ), v.detach()

        return

    def trainLifetimeValueAndIntrinsicReward(self, trajectory):
        returns = self.discountedReturnFn(trajectory=trajectory[-2::-1],
                                          gamma=self.outer_gamma,
                                          bootstrap_value=trajectory[-1]["lifetime_value"],
                                          reward_key="extrinsic_reward")

        intrinsic_reward_and_lifeimte_value_loss = 0
        for index in range(len(trajectory) - 1):
            action_prob, _ = self.policy(
                trajectory[index]["observation"])
            log_prob = Categorical(action_prob).log_prob(
                torch.tensor(trajectory[index]["action"]))
            lifetime_value = trajectory[index]["lifetime_value"]
            advantage = (returns[index] - lifetime_value)

            policy_loss = (self.inner_gamma ** index) * \
                advantage.detach() * log_prob
            value_loss = F.smooth_l1_loss(returns[index], lifetime_value)
            intrinsic_reward_and_lifeimte_value_loss += -policy_loss + value_loss

        self.intrinsic_reward_and_lifeimte_value_optimizer.zero_grad()
        intrinsic_reward_and_lifeimte_value_loss.backward()
        self.intrinsic_reward_and_lifeimte_value_optimizer.step()

        # fix policy
        for _, module in self.policy.named_children():
            for param_key in module._parameters:
                param = module._parameters[param_key]
                module._parameters[param_key] = param.detach(
                ).requires_grad_(True)

        return

    def prepareForLife(self):
        # life preparations:
        self.seed = np.random.randint(1, 1000)
        self.env_observation, self.env_info = self.env.reset(seed=self.seed)
        print(self.env_observation, self.env_info)
        # FIXME: this is not random
        self.policy = Policy(state_space=self.state_space,
                             action_space=self.action_space)
        self.episode_count = 0
        self.episode_step_count = 0
        self.lifetime_done = False
        self.episode_rewards.clear()
        self.episode_returns.clear()

        self.intrinsic_reward_and_lifeimte_value_h0 = None

    def trainLife(self):
        # global preparations:

        for i in count(1):
            self.prepareForLife()

            while not self.lifetime_done:  # outer loop
                # step1: sample a new task and a new random policy parameter

                trajectories = []

                # step2: update policy (inner loop)
                for _ in range(self.outer_unroll_length):
                    # 1) generate a trajectory
                    trajectory = self.generateATrajectory()
                    # 2) append it to trajectories
                    trajectories += trajectory
                    # 3) feed to RNN
                    self.feedToRNN(trajectory)

                    # if there is only 1 transition, only add it to lifetime
                    # that happens at the last transition. -> terminated = True
                    if len(trajectory) == 1:
                        continue
                    # 4) train, update policy
                    self.trainTrajectory(trajectory)

                # step3: update intrinsic reward function
                # step4: update lifetime value function
                self.trainLifetimeValueAndIntrinsicReward(trajectories)

                print("episode {} \t latest return {:.2f} \t average return {:.2f}".format(
                    self.episode_count, self.episode_returns[-1] if len(self.episode_returns) > 0 else 0, np.average(self.episode_returns) if len(self.episode_returns) > 0 else 0))

            plt.plot(self.episode_returns)
            plt.show()


def main():
    cartPole = CartPole()
    agent = Agent(cartPole)

    agent.trainLife()


if __name__ == '__main__':
    main()
