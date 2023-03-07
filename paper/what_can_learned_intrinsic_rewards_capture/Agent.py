from collections import deque

import numpy as np
import torch.optim as optim
from envs.EmptyRooms import EmptyRooms
from models.IntrinsicReward import IntrinsicReward
from models.Policy import Policy


class Agent():
    def __init__(self, env: EmptyRooms):
        # enviornment
        self.env = env

        # some parameters
        self.episode_max_num = 10000  # max iteration in episode
        self.episode_num = 100  # episode number -> N in paper
        self.lifetime_value_step = 10  # n-step trajectory for intrinsic reward udpate

        # misc
        self.policy = Policy(state_space=2, action_space=4)
        self.intrinsic_rewrd = IntrinsicReward(state_space=2)
        self.lifetime_value = IntrinsicReward(state_space=2)
        self.gamma = 1.0
        self.h0 = None

        # saved lifetime
        self.lifetime_trajectory = []

        # optimizer for different parameters
        self.policy_optimizer = optim.Adam(self.policy.parameters())
        self.intrinsic_rewrd_optimizer = optim.Adam(
            self.intrinsic_rewrd.parameters())
        self.lifetime_value_optimizer = optim.Adam(
            self.lifetime_value.parameters())

    def generateEpisodeTrajectory(self, seed):
        # reset the env
        observation, info = self.env.reset(seed=seed)

        trajectory = []

        # prevent repeat forever (avoid epsidoe too long)
        for _ in range(self.episode_max_num):
            action, saved_log = self.policy(observation["agent"])
            next_observation, extrinsic_reward, terminated, _, info = self.env.step(
                action)

            # record in trajctory
            # XXX: Make it elegant.
            trajectory.append({"observation": observation["agent"],
                               "action": action,
                               "saved_log": saved_log,
                               "extrinsic_reward": extrinsic_reward,
                               "intrinsic_reward": self.intrinsic_rewrd((observation["agent"], 0, 0, 0), self.h0)[0],
                               "terminated": 1 if terminated else 0,
                               })

            # prepare for next iteration
            observation = next_observation

            if terminated:
                break

        # # add to lifetime
        # self.lifetime_trajectory += trajectory

        return trajectory

    # train episode (inner loop) and it only updates policy
    def trainEpisode(self, trajectory):
        # update agent/policy
        objective = 0
        stateReturn = 0
        discounted = 1

        # TODO: Replace "tuple" with a better name
        for tuple in trajectory[::-1]:
            saved_log = tuple["saved_log"]
            intrinsic_reward = tuple["intrinsic_reward"]

            stateReturn += discounted * intrinsic_reward
            objective += saved_log * stateReturn.detach()

            discounted = discounted * self.gamma

        # XXX: Probably need to clarify if the objective should be updated with a std...? like REINFORCE
        # returns = (returns - returns.mean()) / (returns.std() + eps)

        # update policy
        self.policy_optimizer.zero_grad()
        objective.backward()
        self.policy_optimizer.step()

        return

    def trainLifetimeValueAndIntrinsicReward(self):
        # update intrinsic reward
        objective_lifetime_value = 0
        objective_intrinsic_reward = 0

        sliding_window = deque()

        # prepare a sliding window
        for index in range(self.lifetime_value_step):
            sliding_window.append()

        for index in range(self.lifetime_trajectory):
            return_life = 0

            # TODO: obviously it could be optimized (not sure if pytorch supports it)
            for index, tuple in enumerate(sliding_window):
                return_life += (self.gamma ** index) * \
                    tuple["extrinsic_reward"]

            # if permits, add a lifetime value
            if index + self.lifetime_value_step <= len(self.lifetime_trajectory):
                assert (len(sliding_window) == self.lifetime_value_step)

                output_n, self.h0 = self.lifetime_value(
                    (sliding_window[-1]["observation"], np.zeros(1,), np.zeros(1,), np.zeros(1,)))

                return_life += (self.gamma **
                                self.lifetime_value_step) * output_n.item()

            objective_lifetime_value += return_life.detach() * \
                sliding_window[0]["saved_log"]

            # calculate intrinsic reward
            output, _ = self.lifetime_value(sliding_window[0]["observation"])
            objective_intrinsic_reward += output.item() * (return_life - output.item().detach())

            # maintain the sliding window
            if index + self.lifetime_value_step < len(self.lifetime_trajectory):
                tuple = self.lifetime_trajectory[index +
                                                 self.lifetime_value_step]
                sliding_window.append(tuple)

            sliding_window.popleft()

        # update intrinsic reward function
        self.intrinsic_rewrd_optimizer.zero_grad()
        -objective_intrinsic_reward.backward()
        self.intrinsic_rewrd_optimizer.step()

        self.lifetime_value_optimizer.zero_grad()
        -objective_lifetime_value.backward()
        self.lifetime_value_optimizer.step()

        return

    def trainLife(self):
        while True:
            # preparations:
            # 1) clear the lifetime trajectory
            self.lifetime_trajectory.clear()

            # step1: sample a new task and a new random policy parameter
            # FIXME: this is not random
            self.policy = Policy(state_space=4, action_space=2)
            seed = np.random.randint(1, 1000)

            # step2: update policy
            for _ in range(self.episode_num):
                trajectory = self.generateEpisodeTrajectory(self, seed)
                self.trainEpisode(trajectory)

                # add to lifetime
                self.lifetime_trajectory += trajectory

            # step3: update intrinsic reward function
            # step4: update lifetime value function
            self.trainLifetimeValueAndIntrinsicReward()
