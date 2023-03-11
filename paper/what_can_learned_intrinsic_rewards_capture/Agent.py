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
        self.episode_num = 1  # episode number -> N in paper
        self.lifetime_value_step = 2  # n-step trajectory for intrinsic reward udpate

        # misc
        self.policy = Policy(state_space=2, action_space=4)
        self.intrinsic_rewrd = IntrinsicReward(input_space=5)
        self.lifetime_value = IntrinsicReward(input_space=5)
        self.gamma = 1.0
        self.intrinsic_reward_h0 = None
        self.lifetime_value_h0 = None

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
            action = action.item()
            next_observation, extrinsic_reward, terminated, _, info = self.env.step(
                action)

            # record in trajctory
            # XXX: Make it elegant.
            # RNN
            terminated = 1 if terminated else 0
            intrinsic_reward, self.intrinsic_reward_h0 = self.intrinsic_rewrd(
                (observation["agent"], action, extrinsic_reward, terminated), self.intrinsic_reward_h0)
            lifetime_value, self.lifetime_value_h0 = self.lifetime_value(
                (observation["agent"], action, extrinsic_reward, terminated), self.lifetime_value_h0)

            # get policy parameters
            self.policy_optimizer.zero_grad()
            saved_log.backward(retain_graph=True)

            trajectory.append({"observation": observation["agent"],
                               "action": action,
                               "saved_log": saved_log,
                               "extrinsic_reward": extrinsic_reward,
                               "intrinsic_reward": intrinsic_reward,
                               "lifetime_value": lifetime_value,
                               "terminated": terminated,
                               "policy_named_parameters": {name: w for name, w in self.policy.named_parameters()}
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
            # float, no need of detach()
            intrinsic_reward = tuple["intrinsic_reward"]

            stateReturn += discounted * intrinsic_reward.item()
            objective += saved_log * stateReturn

            discounted = discounted * self.gamma

        # XXX: Probably need to clarify if the objective should be updated with a std...? like REINFORCE
        # returns = (returns - returns.mean()) / (returns.std() + eps)

        # update policy
        self.policy_optimizer.zero_grad()
        objective = -objective
        objective.backward(retain_graph=True)
        self.policy_optimizer.step()

        return

    def trainLifetimeValueAndIntrinsicReward(self):
        # update intrinsic reward
        objective_lifetime_value = 0
        objective_intrinsic_reward = 0

        sliding_window = deque()

        # prepare a sliding window
        for index in range(min(self.lifetime_value_step, len(self.lifetime_trajectory))):
            sliding_window.append(self.lifetime_trajectory[index])

        for outer_index in range(len(self.lifetime_trajectory)):
            return_life = 0
            objective_intrinsic_reward = 0

            assert (len(sliding_window) > 0)

            policy_named_parameters_0 = sliding_window[0]["policy_named_parameters"]

            # TODO: obviously it could be optimized (not sure if pytorch supports it)
            for inner_index, tuple in enumerate(sliding_window):
                return_life += (self.gamma ** inner_index) * \
                    tuple["extrinsic_reward"]

                # !!!get the objective_intrinsic_reward!!!
                for name, W in tuple["policy_named_parameters"].items():

                    objective_intrinsic_reward += (self.gamma ** inner_index) * tuple["intrinsic_reward"] * (
                        policy_named_parameters_0[name].grad * W.grad).sum().item()

            # if it permits, add a lifetime value
            if outer_index + self.lifetime_value_step < len(self.lifetime_trajectory):
                assert (len(sliding_window) == self.lifetime_value_step)

                tuple_n = self.lifetime_trajectory[outer_index +
                                                   self.lifetime_value_step]
                return_life += (self.gamma **
                                self.lifetime_value_step) * tuple_n["lifetime_value"].item()
                sliding_window.append(tuple_n)

            sliding_window.popleft()

            # calculate objective functions
            # 1) calculate objective lifetime value
            lifetime_value = tuple["lifetime_value"]
            objective_lifetime_value += (return_life -
                                         lifetime_value.item()) * lifetime_value

            # 2) calculate objective intrinsic reward
            objective_intrinsic_reward = return_life * objective_intrinsic_reward

        # update intrinsic reward function
        self.intrinsic_rewrd_optimizer.zero_grad()
        objective_intrinsic_reward = -objective_intrinsic_reward
        objective_intrinsic_reward.backward()
        self.intrinsic_rewrd_optimizer.step()

        self.lifetime_value_optimizer.zero_grad()
        objective_lifetime_value = -objective_lifetime_value
        objective_lifetime_value.backward()
        self.lifetime_value_optimizer.step()

        return

    def trainLife(self):
        while True:
            # preparations:
            # 1) clear the lifetime trajectory
            self.lifetime_trajectory.clear()

            # step1: sample a new task and a new random policy parameter
            # FIXME: this is not random
            self.policy = Policy(state_space=2, action_space=4)
            seed = np.random.randint(1, 1000)

            print("step2")
            # step2: update policy
            for _ in range(self.episode_num):
                # 1) generate trajectory
                trajectory = self.generateEpisodeTrajectory(seed)
                # 2) add to lifetime
                self.lifetime_trajectory += trajectory
                # 3) train episode, update policy
                self.trainEpisode(trajectory)

            print("step3 + 4")
            # step3: update intrinsic reward function
            # step4: update lifetime value function
            self.trainLifetimeValueAndIntrinsicReward()

            print("Train one round")


def main():
    emptyRoom = EmptyRooms(render_mode=None)
    agent = Agent(emptyRoom)

    agent.trainLife()


if __name__ == '__main__':
    main()