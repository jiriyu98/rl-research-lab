from itertools import count

import numpy as np
import torch
import torch.optim as optim
from envs.EmptyRooms import EmptyRooms
from matplotlib import pyplot as plt
from models.IntrinsicReward import IntrinsicReward
from models.Policy import Policy
from torch.distributions import Categorical

'''
Actor-Critic style
'''
torch.autograd.set_detect_anomaly(True)


class Agent():
    def __init__(self, env: EmptyRooms):
        # enviornment
        self.env = env

        # model
        self.policy = Policy(state_space=2, action_space=4)
        self.intrinsic_reward = IntrinsicReward(input_space=5)
        self.lifetime_value = IntrinsicReward(input_space=5)

        # hyperparameters
        self.episode_step_limit = 100
        self.episode_num_per_lifetime = 100
        self.lifetime_done = False
        self.trajectory_length = 8  # n-step trajectory for intrinsic reward udpate
        self.outer_gamma = 0.99
        self.inner_gamma = 0.9
        self.outer_unroll_length = 5
        self.intrinsic_reward_h0 = None
        self.lifetime_value_h0 = None
        self.seed = None

        # utils
        self.adam_count = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_distances = []
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
            list(self.lifetime_value.parameters()) + list(self.intrinsic_reward.parameters()))

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

        for _ in range(self.trajectory_length + 1):
            action_prob, _ = self.policy(observation["agent"])
            action = self.selectAction(action_prob)
            next_observation, extrinsic_reward, terminated, _, info = self.env.step(
                action)
            self.episode_distances.append(info["distance"])
            self.episode_rewards.append(extrinsic_reward)
            self.episode_step_count += 1

            if self.episode_step_count >= self.episode_step_limit:
                terminated = True

            trajectory.append({"observation": observation["agent"],
                               "action": action,
                               "extrinsic_reward": extrinsic_reward,
                               "terminated": terminated,
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

        if self.episode_count >= self.episode_num_per_lifetime:
            self.lifetime_done = True

        self.env_observation = observation
        self.env_info = info

        return trajectory

    def discountedReturnFn(self, trajectory, gamma, bootstrap_value, return_key):
        # Args:
        # trajectory: T + 1 length
        # return_key: can be "extrinsic_reward" or "intrinsic_reward"

        returns = []
        trajectory_return = bootstrap_value

        for index in range(len(trajectory[::-1])):
            trajectory_return = trajectory_return * \
                gamma + trajectory[index][return_key]
            returns.append(trajectory_return)

        return returns[::-1]

    def trainTrajectory(self, trajectory):
        # Args:
        # trajectory: T + 1 length

        # step1) update RNN and state_value, prepare for policy udpate
        for tuple in trajectory:
            observation = tuple["observation"]
            extrinsic_reward = tuple["extrinsic_reward"]
            action = tuple["action"]
            terminated = tuple["terminated"]

            # get intrinsic_reward and lifetime_value
            intrinsic_reward, self.intrinsic_reward_h0 = self.intrinsic_reward(
                (observation, action, extrinsic_reward, terminated), self.intrinsic_reward_h0)
            lifetime_value, self.lifetime_value_h0 = self.lifetime_value(
                (observation, action, extrinsic_reward, terminated), self.lifetime_value_h0)

            _, state_value = self.policy(observation)
            # add intrinsic_reward and lifetime_value into tuple
            tuple["intrinsic_reward"] = intrinsic_reward
            tuple["lifetime_value"] = lifetime_value
            tuple["state_value"] = state_value

        # step2) update policy
        returns = self.discountedReturnFn(trajectory=trajectory[:-1],
                                          gamma=self.inner_gamma,
                                          bootstrap_value=trajectory[-1]["state_value"].item(
        ),
            return_key="intrinsic_reward")

        policy_and_value_objective = 0
        for index in range(self.trajectory_length):
            action_prob, state_value = self.policy(
                trajectory[index]["observation"])
            log_prob = Categorical(action_prob).log_prob(
                torch.tensor(trajectory[index]["action"]))

            advantage = (returns[index] - state_value.item())

            policy_loss = (self.inner_gamma ** index) * advantage * log_prob
            value_loss = advantage.detach() * state_value
            # TODO: Wrong here.
            policy_and_value_objective += -policy_loss + value_loss

        # Version 2
        # non-leaf nodes also need gradient
        # policy_and_value_objective.retain_grad()

        # recomend to use autograd.grad ..?
        # UserWarning: Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this. If you have to use this function, make sure to reset the .grad fields of your parameters to None after use to break the cycle and avoid the leak.
        # policy_and_value_objective.backward(create_graph=True)

        # for module_name, module in self.policy.named_children():
        #     clip_grad_norm_(module.parameters(), 0.2)
        #     for param_key in module._parameters:
        #         param = module._parameters[param_key]
        #         grad = param.grad

        #         print(grad.shape)

        #         self.adam_count += 1
        #         correct1 = 1 - .9 ** self.adam_count
        #         correct2 = 1 - .999 ** self.adam_count

        #         lr = 1e-3
        #         m, v = self.adam_m[module_name][param_key], self.adam_v[module_name][param_key]
        #         m = m + (grad - m) * (1 - .9)
        #         # detach()...? the previous work did this operation
        #         v = v + torch.square(grad.detach() - v) * (1 - .999)
        #         assert (torch.all(v >= 0))
        #         module._parameters[param_key] = param - \
        #             m / correct1 * lr / (torch.sqrt(v / correct2) + 1e-5)

        #         print(module._parameters[param_key].shape)

        # Version 1

        # grad will not be in the variable now
        policy_parameter_grads = torch.autograd.grad(
            policy_and_value_objective, self.policy.parameters(), create_graph=True)

        # clip_grad_norm_(policy_parameter_grads, 0.2)

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

                lr = 1e-3
                m, v = self.adam_m[module_name][param_key], self.adam_v[module_name][param_key]
                m = m + (grad - m) * (1 - .9)
                # detach()...? the previous work did this operation
                v = v + torch.square(grad.detach() - v) * (1 - .999)
                module._parameters[param_key] = param - \
                    m / correct1 * lr / (torch.sqrt(v / correct2) + 1e-5)

        return

    def trainLifetimeValueAndIntrinsicReward(self, trajectory):
        returns = self.discountedReturnFn(trajectory=trajectory[:-1],
                                          gamma=self.outer_gamma,
                                          bootstrap_value=trajectory[-1]["lifetime_value"],
                                          return_key="extrinsic_reward")

        intrinsic_reward_and_lifeimte_value_loss = 0
        for index in range(len(trajectory[:-1])):
            action_prob, _ = self.policy(
                trajectory[index]["observation"])
            log_prob = Categorical(action_prob).log_prob(
                torch.tensor(trajectory[index]["action"]))
            lifetime_value = trajectory[index]["lifetime_value"]

            advantage = (returns[index].item() - lifetime_value.item())

            policy_loss = (self.inner_gamma ** index) * advantage * log_prob
            value_loss = advantage * lifetime_value
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
        self.policy = Policy(state_space=2, action_space=4)
        self.episode_count = 0
        self.episode_step_count = 0
        self.lifetime_done = False
        self.episode_rewards.clear()
        self.episode_returns.clear()

        self.intrinsic_reward_h0 = None
        self.lifetime_value_h0 = None

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
                    # 3) train, update policy
                    self.trainTrajectory(trajectory)

                # step3: update intrinsic reward function
                # step4: update lifetime value function
                self.trainLifetimeValueAndIntrinsicReward(trajectories)

                print("episode {} \t latest return {:.2f} \t average dis {:.2f}".format(
                    self.episode_count, self.episode_returns[-1] if len(self.episode_returns) > 0 else 0, np.average(self.episode_distances) if len(self.episode_returns) > 0 else 0))

            plt.plot(self.episode_returns)
            plt.show()


def main():
    emptyRoom = EmptyRooms(render_mode="human")
    # emptyRoom = EmptyRooms(render_mode=None)
    agent = Agent(emptyRoom)

    agent.trainLife()


if __name__ == '__main__':
    main()
