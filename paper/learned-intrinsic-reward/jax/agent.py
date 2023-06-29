import argparse
import pickle
from collections import namedtuple
from functools import partial

import coax
import gymnasium as gym
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from jax.tree_util import register_pytree_node
from rlax import (discounted_returns, entropy_loss, lambda_returns,
                  policy_gradient_loss)

# config.update("jax_debug_nans", True)

# ref: https://github.com/hamishs/JAX-RL/blob/main/src/jax_rl/algorithms/ppo.py
# ref: https://github.com/deepmind/dm-haiku/blob/f6439a1a234f1c46f0d515ade42de6f47553e7ef/examples/impala/learner.py#L38
# ref: https://github.com/deepmind/dm-haiku/blob/main/examples/impala/learner.py


# Create the parser
parser = argparse.ArgumentParser(description='Learned Intrinsic Reward Agent')

# Add arguments
parser.add_argument(
    '--heat_map_output', default='./heatmap/', help='Path to the the output heatmap directory')
parser.add_argument(
    '--episode_num_per_lifetime', default=200, type=int, help='')
parser.add_argument(
    '--multiple_lifetime_mode', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument(
    '--save_heatmap_per_time', default=100, type=int, help='')
parser.add_argument(
    '--max_episode_steps', default=20, type=int, help='')

# Parse the command-line arguments
args = parser.parse_args()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Learned-Intrinsic-Reward",

    # track hyperparameters and run metadata
    config={
        "architecture": "FF",
        "heat_map_output": args.heat_map_output,
        "episode_num_per_lifetime": args.episode_num_per_lifetime,
        "multiple_lifetime_mode": args.multiple_lifetime_mode,
        "max_episode_steps": args.max_episode_steps,
    }
)


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


TrainState = namedtuple(
    "TrainState", ["params", "opt_state"])
register_pytree_node(
    TrainState,
    lambda xs: (tuple(xs), None),
    lambda _, xs: TrainState(*xs)
)


class Agent():
    def __init__(self, env, lr_policy, lr_irs, buffer_size=20, buffer_batch_size=5) -> None:
        # prng
        self.prng = hk.PRNGSequence(2)

        # policy and irs
        def policy_forward(S):
            logits = hk.Sequential((
                hk.Linear(64, w_init=None), jax.nn.relu,
                hk.Linear(env.action_space.n, w_init=None),
            ))
            values = hk.Sequential((
                hk.Linear(64, w_init=None), jax.nn.relu,
                hk.Linear(1, w_init=None), jnp.ravel
            ))
            return logits(S), values(S)

        policy = hk.without_apply_rng(hk.transform(policy_forward))

        policy_params = policy.init(
            rng=next(self.prng), S=jnp.ones([1, env.observation_space.n]))
        self.policy_transform = policy
        self.policy = policy.apply
        self.policy_update, policy_opt_state = self.init_optimiser(
            lr_policy, policy_params)

        def irs_forward(S):
            values1 = hk.Sequential((
                hk.Linear(64, w_init=None), jax.nn.relu,
                hk.Linear(1, w_init=None),
                jnp.arctan, jnp.ravel
            ))
            values2 = hk.Sequential((
                hk.Linear(64, w_init=None), jax.nn.relu,
                hk.Linear(1, w_init=None),
                jnp.ravel
            ))
            return values1(S), values2(S)

        irs = hk.without_apply_rng(hk.transform(irs_forward))

        irs_params = irs.init(
            rng=next(self.prng), S=jnp.ones([1, env.observation_space.n]))
        self.irs_transform = irs
        self.irs = irs.apply
        self.irs_update, irs_opt_state = self.init_optimiser(
            lr_irs, irs_params)

        # trainstate
        self.policy_learner_state = TrainState(
            policy_params, policy_opt_state)
        self.irs_learner_state = TrainState(
            irs_params, irs_opt_state
        )

        # env
        self.env = env

        # buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_size, batch_size=buffer_batch_size, truncated=False)

        # gamma
        self.ep_gamma = 0.9
        self.life_gamma = 0.99

    # Note that this method should be pure function
    def inner_update(self, eta, learner_state: TrainState, rollout):
        def inner_loss(theta, rollout, returns, advantages):
            logits, v = self._apply_param_on_obs(
                self.policy, theta, rollout["s"])
            masks = jnp.ones_like(advantages)
            pg_loss = policy_gradient_loss(logits_t=logits,
                                           a_t=rollout["a"],
                                           adv_t=advantages,
                                           w_t=masks,
                                           use_stop_gradient=False,)  # important to keep gradient
            baseline_loss = 0.5 * \
                jnp.mean(jnp.square(v - returns) * masks)
            entro_loss = entropy_loss(logits_t=logits, w_t=masks)

            return pg_loss + 0.5 * baseline_loss + 0.01 * entro_loss

        policy_params = learner_state.params
        policy_opt_state = learner_state.opt_state

        irs, _ = self._apply_param_on_obs(
            self.irs, eta, rollout["ns"])

        # This is the line to split the episode
        # when lambda is zero, n-step bootstrapping will stop at the end of the episode
        # See more at:
        # https://github.com/deepmind/dm-haiku/blob/adfb8331b590fa036e19f7ee58b9e1d5a2a8d189/examples/impala/learner.py#L114C1-L117C36
        lambda_ = jnp.not_equal(rollout["done"], True).astype(jnp.float32)

        returns = lambda_returns(
            r_t=irs,
            discount_t=rollout["discounted_t"] * self.ep_gamma,
            # NOTE: Rollout, Episode, Trajectory. In deepmind context, it can cross multiple episodes
            # See more at:
            # https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#trajectories
            # https://github.com/deepmind/dm-haiku/blob/main/examples/impala/learner.py
            v_t=rollout["nv"],
            lambda_=lambda_,
            stop_target_gradients=False,
        )
        advantages = returns - rollout["v"]

        dldtheta = jax.grad(inner_loss)(
            policy_params, rollout, returns, advantages)
        # p_updates, policy_opt_state = self.policy_update(
        #     dldtheta, policy_opt_state, policy_params)
        # policy_params = optax.apply_updates(policy_params, p_updates)

        policy_params = jax.tree_map(
            lambda g, state: state - 0.01 * g, dldtheta, policy_params)

        return TrainState(policy_params, policy_opt_state)

    def outer_update(self, irs_learner_state: TrainState, policy_learner_state: TrainState, rollouts):
        # namedtuple can be used to optimize
        eta = irs_learner_state.params
        irs_opt_state = irs_learner_state.opt_state

        def outer_loss(eta, learner_state, rollouts):
            all_ex_r = []
            all_action = []
            all_logits = []
            all_v_lifetime = []
            all_discounts = []

            for rollout in rollouts:
                # learner_state = self.inner_update(
                #     eta, learner_state, rollout)

                all_ex_r.append(rollout["ex_r"])
                all_action.append(rollout["a"])
                all_discounts.append(rollout["discounted_t"])

                logits, _ = self._apply_param_on_obs(
                    self.policy, learner_state.params, rollout["s"])
                all_logits.append(logits)

                _, lifetime_value = self._apply_param_on_obs(
                    self.irs, eta, rollout["s"])
                all_v_lifetime.append(lifetime_value)

                _, bootstrap_value_v_lifetime = self._apply_param_on_obs(
                    self.irs, eta, rollout["ns"][-1])

                learner_state = self.inner_update(
                    eta, learner_state, rollout)

            all_ex_r = jnp.concatenate(all_ex_r)
            all_action = jnp.concatenate(all_action)
            all_logits = jnp.concatenate(all_logits)
            all_discounts = jnp.concatenate(all_discounts)
            all_v_lifetime = jnp.concatenate(all_v_lifetime)

            returns = discounted_returns(
                r_t=all_ex_r,
                discount_t=all_discounts * self.life_gamma,
                v_t=bootstrap_value_v_lifetime,
            )
            advantages = returns - all_v_lifetime
            masks = jnp.ones_like(advantages)

            pg_loss = policy_gradient_loss(
                logits_t=all_logits,
                a_t=all_action,
                adv_t=advantages,
                w_t=masks,
                use_stop_gradient=True,)
            baseline_loss = 0.5 * \
                jnp.mean(jnp.square(all_v_lifetime - returns) * masks)
            entro_loss = entropy_loss(
                logits_t=all_logits, w_t=masks)

            return pg_loss + 0.5 * baseline_loss + 0.01 * entro_loss

        value, dmdeta = jax.value_and_grad(outer_loss)(
            eta, policy_learner_state, rollouts)
        eta_updates, irs_opt_state = self.irs_update(
            dmdeta, irs_opt_state, eta)
        irs_params = optax.apply_updates(eta, eta_updates)

        # wandb log
        wandb.log({
            "irs_loss": value.item(),
            "avg_G": self.env.avg_G,
            "episode_num": self.env.ep,
            "transitions": self.env.T})

        return TrainState(irs_params, irs_opt_state)

    def _observation_preprocess(self, X):
        X = jnp.asarray(X)
        X = jax.nn.one_hot(X, self.env.observation_space.n)
        X = jnp.reshape(X, (-1, self.env.observation_space.n))
        return X

    def _apply_param_on_obs(self, model, theta, s):
        X = self._observation_preprocess(s)
        return model(theta, X)

    def _sample(self, logits):
        return int(jax.random.choice(next(self.prng), self.env.action_space.n, p=logits))

    def _sample_func(self, obs):
        logits, _ = self._apply_param_on_obs(
            self.policy, self.policy_learner_state.params, obs)
        logits = jax.nn.softmax(logits).squeeze()
        X = self._sample(logits)

        return X, logits[X]

    def train(self):
        policy_learner_state = self.policy_learner_state
        rollouts = self.buffer.get_rollouts()

        # update at the beginning because outer_update won't update policy_learner_state
        for rollout in rollouts:
            eta = self.irs_learner_state.params
            policy_learner_state = self.inner_update(
                eta, policy_learner_state, rollout)

        # then
        self.irs_learner_state = self.outer_update(
            self.irs_learner_state, self.policy_learner_state, rollouts)
        self.policy_learner_state = policy_learner_state

    def learn(self):
        # for tracking and plot
        intrinsic_update_count = 0

        for ep in range(1, 50_0000):
            # lifetime reset
            if args.multiple_lifetime_mode and ep % args.episode_num_per_lifetime == 0:
                # reset env
                self.env = coax.wrappers.TrainMonitor(
                    gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=False))
                # reset policy params
                self.policy_params = self.policy_transform.init(
                    rng=next(self.prng), S=jnp.ones([1, self.env.observation_space.n]))

            s, _ = self.env.reset()

            for t in range(1, self.env.spec.max_episode_steps):
                a, logp = self._sample_func(s)
                s_next, ex_r, done, truncated, _ = self.env.step(a)

                # small incentive to keep moving
                if jnp.array_equal(s_next, s):
                    ex_r = -0.01

                _, v = self._apply_param_on_obs(
                    self.policy, self.policy_learner_state.params, s)
                _, nv = self._apply_param_on_obs(
                    self.policy, self.policy_learner_state.params, s_next)
                self.buffer.push(
                    s, a, ex_r, v[0], nv[0], s_next, done)

                if self.buffer.is_full():
                    # update counter
                    intrinsic_update_count += 1
                    # train
                    self.train()

                    if intrinsic_update_count % args.save_heatmap_per_time == 0:
                        # save params and heatmap
                        self.save(self.policy_learner_state,
                                  './tmp/params/policy.dp')
                        self.save(self.irs_learner_state,
                                  './tmp/params/irs.dp')
                        self.save_heatmap(
                            name="heatmap_after_{}_updates".format(
                                intrinsic_update_count),
                            path=args.heat_map_output)

                    self.buffer.reset()  # then reset the buffer

                if done or truncated:
                    break

                s = s_next

            # early stopping
            if not args.multiple_lifetime_mode and self.env.avg_G > self.env.spec.reward_threshold:
                break

    def init_optimiser(self, lr, params):
        optimizer = optax.chain(
            # optax.clip(1.0),
            optax.adam(lr, b1=0.9, b2=0.99),
        )
        opt_state = optimizer.init(params)
        return optimizer.update, opt_state

    def save(self, state, path):
        with open(path, 'wb') as fp:
            pickle.dump(state, fp)

    def load(self, path) -> hk.Params:
        with open(path, 'rb') as fp:
            return pickle.load(fp)

    def save_heatmap(self, name="temp", path="./heatmap/", **kwargs):
        plt.figure(figsize=[12, 12])
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # 1. task

        desc = self.env.env.desc.copy().tolist()
        desc_dict = {b'S': 0, b'F': 1, b'H': 2, b'G': 3}
        m, n = len(desc), len(desc[0])
        for i in range(m):
            for j in range(n):
                desc[i][j] = desc_dict[desc[i][j]]
        desc = np.array(desc)
        ax = plt.subplot(231)
        self.generate_heatmap("task env", desc, ax)

        # 2&3. irs and lifetime_ex_r
        desc = jnp.arange(16)
        irs, lifetime_ex_r = self._apply_param_on_obs(
            self.irs, self.irs_learner_state.params, desc)
        irs = irs.reshape(m, n)
        lifetime_ex_r = lifetime_ex_r.reshape(m, n)
        ax = plt.subplot(232)
        self.generate_heatmap("irs heatmap", irs, ax)

        ax = plt.subplot(233)
        self.generate_heatmap(
            "estimated lifetime-ex reward", lifetime_ex_r, ax)

        # 4. irs_value
        _, v = self._apply_param_on_obs(
            self.policy, self.policy_learner_state.params, desc)
        v = v.reshape(m, n)
        ax = plt.subplot(234)
        self.generate_heatmap("estimated irs value", v, ax)

        # 5. rollout exps
        rollouts = self.buffer.get_rollouts()
        desc = [0] * (m * n)
        for rollout in rollouts:
            for i in range(len(rollout)):
                s, ns, ex_r = rollout["s"][i], rollout["ns"][i], rollout["ex_r"][i]
                desc[int(s)] += 1
                if ex_r > 0:
                    desc[int(ns)] += 1

        desc = jnp.asarray(desc).reshape(m, n)
        ax = plt.subplot(235)
        self.generate_heatmap("the last rollouts visit count", desc, ax)

        # 6. rollout exps

        # save as png
        plt.savefig(path + name)
        plt.clf()
        plt.close()

    def generate_heatmap(self, name, data, ax, col_labels=None, row_labels=None):
        im = ax.imshow(data, cmap="PuBuGn")
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(None, rotation=-90, va="bottom")
        ax.set_title(name)
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


class ReplayBuffer(object):
    def __init__(self, capacity, batch_size, truncated=False):
        self.capacity = capacity  # transition capacity
        self.batch_size = batch_size  # every batch max trasition capacity
        self.max_len = self.capacity // self.batch_size  # buffer size of rollout
        self.truncated = truncated
        self.reset()

    def push(self, state, action, reward, value, next_value, next_state, done):
        rollout = self.buffer[self.trajectory_idx]
        rollout.append((state, action, reward, value,
                        next_value, next_state, done))

        if (len(rollout) == self.batch_size) or (done and self.truncated):
            self.trajectory_idx += 1

    def format_trajectory(self, trajectory):
        state, action, reward, value, next_value, next_state, done = zip(
            *trajectory)
        return {'s': jnp.asarray(state), 'a': jnp.asarray(action),
                'ex_r': jnp.asarray(reward),  'v': jnp.asarray(value), 'nv': jnp.asarray(next_value),
                'ns': jnp.asarray(next_state), 'done': jnp.asarray(done),
                'discounted_t': jnp.ones_like(jnp.asarray(state))}

    def get_rollouts(self):
        rollouts = []
        for trajectory in self.buffer:
            rollouts.append(self.format_trajectory(trajectory))

        return rollouts

    def is_full(self):
        return self.trajectory_idx == self.max_len

    def reset(self):
        self.buffer = [[] for _ in range(self.max_len)]
        self.trajectory_idx = 0


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1',
                   desc=generate_random_map(
                       size=4) if args.multiple_lifetime_mode else ["SFFF", "FFFH", "FFFH", "HFFG"],
                   is_slippery=False,
                   max_episode_steps=args.max_episode_steps)
    env = coax.wrappers.TrainMonitor(env)
    agent = Agent(env, 0.02, 0.001)

    agent.learn()
