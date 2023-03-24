import gym


class CartPole(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make('CartPole-v1')

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return {"agent": obs}, reward, done, False, info

    def reset(self, seed):
        obs = self.env.reset(seed=seed)
        return {"agent": obs}, None
