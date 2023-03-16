from collections import deque
from itertools import count

import matplotlib.pyplot as plt
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
        self.episode_max_num = 100  # max iteration in episode
        self.episode_num = 200  # episode number -> N in paper
        self.lifetime_value_step = 8  # n-step trajectory for intrinsic reward udpate

        # misc
        self.policy = Policy(state_space=2, action_space=4)
        self.intrinsic_rewrd = IntrinsicReward(input_space=5)
        self.lifetime_value = IntrinsicReward(input_space=5)
        self.gamma = 0.99
        self.intrinsic_reward_h0 = None
        self.lifetime_value_h0 = None

        # optimizer for different parameters
        self.policy_optimizer = optim.Adam(self.policy.parameters())
        self.intrinsic_rewrd_optimizer = optim.Adam(
            self.intrinsic_rewrd.parameters())
        self.lifetime_value_optimizer = optim.Adam(
            self.lifetime_value.parameters())

    def generateEpisodeTrajectory(self, seed):
        # reset the env
        observation, info = self.env.reset(seed=seed)
        self.env.render()
        init_observation = observation
        init_info = info

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

        return trajectory, init_observation, init_info

    # train episode (inner loop) and it only updates policy
    def trainEpisode(self, trajectory):
        # update agent/policy
        objective = 0
        episode_intrinsic_return = 0
        discounted = 1
        episode_extrinsic_return = 0

        # TODO: Replace "tuple" with a better name
        for tuple in trajectory[::-1]:
            saved_log = tuple["saved_log"]
            intrinsic_reward = tuple["intrinsic_reward"]
            extrinsic_reward = tuple["extrinsic_reward"]

            episode_intrinsic_return = discounted * \
                episode_intrinsic_return + intrinsic_reward.item()
            episode_extrinsic_return = discounted * \
                episode_extrinsic_return + extrinsic_reward

            objective += saved_log * episode_intrinsic_return
            discounted = discounted * self.gamma

        # XXX: Probably need to clarify if the objective should be updated with a std...? like REINFORCE
        # returns = (returns - returns.mean()) / (returns.std() + eps)

        # update policy
        self.policy_optimizer.zero_grad()
        objective = -objective
        objective.backward()
        self.policy_optimizer.step()

        return episode_extrinsic_return

    def trainLifetimeValueAndIntrinsicReward(self, trajectory):
        # update intrinsic reward
        objective_lifetime_value = 0
        objective_intrinsic_reward = 0

        sliding_window = deque()

        # prepare a sliding window
        for index in range(min(self.lifetime_value_step, len(trajectory))):
            sliding_window.append(trajectory[index])

        for outer_index in range(len(trajectory)):
            return_life = 0

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
            if outer_index + self.lifetime_value_step < len(trajectory):
                assert (len(sliding_window) == self.lifetime_value_step)

                tuple_n = trajectory[outer_index +
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
        # global preparations:

        record_average_return_y = [0] * self.episode_num
        record_average_return_x = [i for i in range(self.episode_num)]

        # life loop
        for i in count(1):
            # step1: sample a new task and a new random policy parameter
            # FIXME: this is not random
            self.policy = Policy(state_space=2, action_space=4)
            seed = np.random.randint(1, 1000)
            # 1) create running_reward for one life
            running_reward = 0

            for j in range(self.episode_num):
                # step2: update policy (episodic)

                # 1) generate trajectory
                trajectory, init_observation, init_info = self.generateEpisodeTrajectory(
                    seed)
                # 2) add to lifetime
                # 3) train episode, update policy
                episode_return = self.trainEpisode(trajectory)

                # 4) log
                running_reward = 0.05 * episode_return + \
                    (1 - 0.05) * running_reward

                # step3: update intrinsic reward function
                # step4: update lifetime value function
                self.trainLifetimeValueAndIntrinsicReward(trajectory)

                if j == 0:
                    print("Init observation {} and init info {}".format(
                        init_observation, init_info))

                print("Life {}, {} episode \t running reward {:.2f}".format(
                    i, j, running_reward))

                record_average_return_y[j] = (
                    record_average_return_y[j] * (i - 1) + episode_return) / i

                if j % 50 == 0:
                    plt.plot(record_average_return_x, record_average_return_y)
                    plt.show()


def main():
    emptyRoom = EmptyRooms(render_mode=None)
    agent = Agent(emptyRoom)

    agent.trainLife()


if __name__ == '__main__':
    main()
