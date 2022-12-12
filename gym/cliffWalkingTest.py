import gym
from gym.utils.env_checker import check_env

env = gym.make('CliffWalking-v0', render_mode="human")
env.unwrapped
check_env(env.unwrapped)

observation, info = env.reset(seed=42)

observation, reward, terminated, truncated, info = env.step(1)
print(reward, terminated)


# alpha = 0.3 # step size
# epsilon = 0.01 # small epsilon


# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

env.close()