import haiku as hk
import jax
import jax.numpy as jnp
from rlax._src.multistep import discounted_returns
from rlax._src.policy_gradients import policy_gradient_loss, entropy_loss
import gymnasium as gym
import optax
import chex
import coax
from unittest import TestCase
import pickle
from functools import partial

# ref: https://github.com/hamishs/JAX-RL/blob/main/src/jax_rl/algorithms/ppo.py


class Agent():
    def __init__(self, env, lr_policy, lr_irs, buffer_size=100, buffer_batch_size=20) -> None:
        # prng
        self.prng = hk.PRNGSequence(2)

        # policy and irs
        def policy_forward(S):
            logits = hk.Sequential(
                (hk.Linear(env.action_space.n, w_init=jnp.zeros), jax.nn.softmax))
            values = hk.Sequential((hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
            return logits(S), values(S)

        policy = hk.without_apply_rng(hk.transform(policy_forward))

        policy_params = policy.init(
            rng=next(self.prng), S=jnp.ones([1, env.observation_space.n]))
        self.policy = policy.apply
        self.policy_update, policy_opt_state = self.init_optimiser(
            lr_policy, policy_params)

        def irs_forward(S):
            values1 = hk.Sequential(
                (hk.Linear(1, w_init=jnp.zeros), jnp.ravel, jnp.tanh))
            values2 = hk.Sequential(
                (hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
            return values1(S), values2(S)

        irs = hk.without_apply_rng(hk.transform(irs_forward))

        irs_params = irs.init(
            rng=next(self.prng), S=jnp.ones([1, env.observation_space.n]))
        self.irs = irs.apply
        self.irs_update, irs_opt_state = self.init_optimiser(
            lr_irs, irs_params)

        # construct state
        self.policy_learner_state = {"params": policy_params,
                                     "opt_state": policy_opt_state,
                                     }
        self.irs_learner_state = {"params": irs_params,
                                  "opt_state": irs_opt_state,
                                  }

        # env
        self.env = env

        # buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_size, batch_size=buffer_batch_size)
        self.buffer_size = buffer_size
        self.buffer_batch_size = buffer_batch_size

        # gamma
        self.ep_gamma = 0.9
        self.life_gamma = 0.99

    # Note that this method should be pure function
    def inner_update(self, eta, learner_state, rollout):
        policy_params = learner_state["params"]
        policy_opt_state = learner_state["opt_state"]

        irs, _ = self._apply_param_on_obs(
            self.irs, eta, rollout["s"])

        returns = discounted_returns(
            r_t=irs,
            discount_t=rollout["discounted_t"] * self.ep_gamma,
            v_t=rollout["v"][-1],
        )
        advantages = returns - rollout["v"]
        masks = jnp.ones_like(advantages)

        def inner_loss(theta, rollout, returns, advantages):
            logits, v = self._apply_param_on_obs(
                self.policy, theta, rollout["s"])
            pg_loss = policy_gradient_loss(logits_t=logits,
                                           a_t=rollout["a"],
                                           adv_t=advantages,
                                           w_t=masks,
                                           use_stop_gradient=True,)
            baseline_loss = 0.5 * jnp.sum(jnp.square(v - returns) * masks)
            entro_loss = entropy_loss(logits_t=logits, w_t=masks)

            return pg_loss + baseline_loss + 0.01 * entro_loss

        dl_dtheta = jax.grad(inner_loss)(
            policy_params, rollout, returns, advantages)
        p_updates, policy_opt_state = self.policy_update(
            dl_dtheta, policy_opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, p_updates)

        return {"params": policy_params,
                "opt_state": policy_opt_state, }

    def outer_update(self, irs_learner_state, policy_learner_state, rollouts):
        # namedtuple can be used to optimize
        eta = irs_learner_state["params"]
        irs_opt_state = irs_learner_state["opt_state"]

        def outer_loss(eta, learner_state, rollouts):
            all_ex_r = []
            all_action = []
            all_logits = []
            all_v_lifetime = []
            all_discounts = []

            theta = learner_state["params"]

            for rollout in rollouts:
                all_ex_r.append(rollout["ex_r"])
                all_action.append(rollout["a"])
                all_discounts.append(rollout["discounted_t"])
                logits, _ = self._apply_param_on_obs(
                    self.policy, theta, rollout["s"])
                irs, lifetime_value = self._apply_param_on_obs(
                    self.irs, eta, rollout["s"])
                all_logits.append(logits)
                all_v_lifetime.append(lifetime_value.squeeze())  # squeeze
                bootstrap_value_v_lifetime = lifetime_value[-1]

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
                use_stop_gradient=False)
            baseline_loss = 0.5 * \
                jnp.sum(jnp.square(all_v_lifetime - returns) * masks)
            entro_loss = entropy_loss(
                logits_t=all_logits, w_t=masks)

            return pg_loss + baseline_loss + 0.01 * entro_loss

        dmdeta = jax.grad(outer_loss)(eta, policy_learner_state, rollouts)
        eta_updates, irs_opt_state = self.irs_update(
            dmdeta, irs_opt_state, eta)
        irs_params = optax.apply_updates(eta, eta_updates)

        return {
            "params": irs_params,
            "opt_state": irs_opt_state
        }

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
            self.policy, self.policy_learner_state["params"], obs)
        logits = logits.squeeze()
        X = self._sample(logits)

        return X, logits[X]

    def train(self):
        rollouts = self.buffer.get_rollouts()
        policy_learner_state = self.policy_learner_state

        # update at the beginning because outer_update won't update policy_learner_state
        for rollout in rollouts:
            eta = self.irs_learner_state["params"]
            policy_learner_state = self.inner_update(
                eta, policy_learner_state, rollout)

        # then
        rollouts = self.buffer.get_rollouts()
        self.irs_learner_state = self.outer_update(
            self.irs_learner_state, self.policy_learner_state, rollouts)
        self.policy_learner_state = policy_learner_state

    def learn(self):
        for ep in range(50_000):
            s, _ = self.env.reset()

            for t in range(1, self.env.spec.max_episode_steps):
                a, logp = self._sample_func(s)
                s_next, ex_r, done, truncated, _ = self.env.step(a)

                # small incentive to keep moving
                if jnp.array_equal(s_next, s):
                    ex_r = -0.01

                _, v = self._apply_param_on_obs(
                    self.policy, self.policy_learner_state["params"], s)
                self.buffer.push(s, a, ex_r, v[0], s_next, done)

                if len(self.buffer) == self.buffer_size:
                    self.train()
                    self.buffer.reset()  # then reset the buffer

                    self.save(self.policy_learner_state,
                              './tmp/params/policy.dp')
                    self.save(self.irs_learner_state,
                              './tmp/params/irs.dp')

                if done or truncated:
                    break

                s = s_next

            # early stopping
            if self.env.avg_G > self.env.spec.reward_threshold:
                break

    def init_optimiser(self, lr, params):
        opt_init, opt_update = optax.adam(lr)
        opt_state = opt_init(params)
        return opt_update, opt_state

    def save(self, state, path):
        with open(path, 'wb') as fp:
            pickle.dump(state, fp)

    def load(self, path) -> hk.Params:
        with open(path, 'rb') as fp:
            return pickle.load(fp)


class ReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.buffer = []
        self.batch_size = batch_size

    def push(self, state, action, reward, value, next_state, done):
        self.buffer.append((state, action, reward, value, next_state, done))

    def get_rollout(self, idx, batch_size):
        state, action, reward, value, next_state, done = zip(
            *self.buffer[idx:idx + batch_size])
        return {'s': jnp.asarray(state), 'a': jnp.asarray(action),
                'ex_r': jnp.asarray(reward),  'v': jnp.asarray(value),
                'ns': jnp.asarray(next_state), 'done': jnp.asarray(done),
                'discounted_t': jnp.ones_like(jnp.asarray(state))}

    def get_rollouts(self):
        idx = 0
        while idx < self.capacity:
            yield self.get_rollout(idx, self.batch_size)
            idx += self.batch_size

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False)
    env = coax.wrappers.TrainMonitor(env)
    agent = Agent(env, 0.01, 0.01)

    agent.learn()
