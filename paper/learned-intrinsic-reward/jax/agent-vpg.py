import haiku as hk
import jax
import jax.numpy as jnp
from rlax._src.multistep import discounted_returns
from rlax._src.policy_gradients import policy_gradient_loss, entropy_loss
import gymnasium as gym
import optax
import coax
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

        self.policy_params = policy.init(
            rng=next(self.prng), S=jnp.ones([1, env.observation_space.n]))
        self.policy = policy.apply
        self.policy_update, self.policy_opt_state = self.init_optimiser(
            lr_policy, self.policy_params)

        def irs_forward(S):
            values1 = hk.Sequential(
                (hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
            values2 = hk.Sequential(
                (hk.Linear(1, w_init=jnp.zeros), jnp.ravel, jnp.tanh))
            return values1(S), values2(S)

        irs = hk.without_apply_rng(hk.transform(irs_forward))

        self.irs_params = irs.init(
            rng=next(self.prng), S=jnp.ones([1, env.observation_space.n]))
        self.irs = irs.apply
        self.irs_update, self.irs_opt_state = self.init_optimiser(
            lr_irs, self.irs_params)

        # env
        self.env = env

        # buffer
        self.buffer = ReplayBuffer(
            capacity=buffer_size, batch_size=buffer_batch_size)
        self.buffer_size = buffer_size
        self.buffer_batch_size = buffer_batch_size

        # gamma
        self.gamma = 0.9

    def inner_update(self, rollout):
        returns = discounted_returns(
            r_t=rollout["ex_r"],
            discount_t=rollout["discounted_t"] * self.gamma,
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

            return pg_loss + 0.5 * baseline_loss + 0.01 * entro_loss

        dl_dtheta = jax.grad(inner_loss)(
            self.policy_params, rollout, returns, advantages)

        p_updates, self.policy_opt_state = self.policy_update(
            dl_dtheta, self.policy_opt_state, self.policy_params)
        self.policy_params = optax.apply_updates(self.policy_params, p_updates)

    def outer_update(self, rollouts):
        for rollout in rollouts:
            self.inner_update(rollout)

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
            self.policy, self.policy_params, obs)
        logits = logits.squeeze()
        X = self._sample(logits)

        return X, logits[X]

    def train(self):
        for ep in range(50_000):
            s, info = self.env.reset()

            for t in range(1, self.env.spec.max_episode_steps):
                a, logp = self._sample_func(s)
                s_next, ex_r, done, truncated, info = self.env.step(a)

                # small incentive to keep moving
                if jnp.array_equal(s_next, s):
                    ex_r = -0.01

                _, v = self._apply_param_on_obs(
                    self.policy, self.policy_params, s)
                self.buffer.push(s, a, ex_r, v[0], s_next, done)

                if len(self.buffer) == self.buffer_size:
                    rollouts = self.buffer.get_rollouts()
                    self.outer_update(rollouts=rollouts)
                    self.buffer.reset()

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


class ReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.buffer = []
        self.batch_size = batch_size

    def push(self, state, action, reward, value, next_state, done):
        self.buffer.append((state, action, reward, value, next_state, done))

    def get_rollout(self, idx):
        state, action, reward, value, next_state, done = zip(
            *self.buffer[idx:idx + self.batch_size])
        return {'s': jnp.asarray(state), 'a': jnp.asarray(action),
                'ex_r': jnp.asarray(reward),  'v': jnp.asarray(value),
                'ns': jnp.asarray(next_state), 'done': jnp.asarray(done),
                'discounted_t': jnp.ones_like(jnp.asarray(state))}

    def get_rollouts(self):
        idx = 0
        while idx < self.capacity:
            yield self.get_rollout(idx)
            idx += self.batch_size

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False)
    env = coax.wrappers.TrainMonitor(env)
    agent = Agent(env, 0.02, 0.01)

    agent.train()
