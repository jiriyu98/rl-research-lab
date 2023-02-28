import gym
from gym import spaces
import pygame
import numpy as np


class EmptyRooms(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5) -> None:
        super().__init__()

        assert (size % 2 == 1)
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # observation_space -> dictionary
        # location is encoded as an element of (x, y)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size * 2, shape=(2,), dtype=int),
                "target": spaces.Box(0, size * 2, shape=(2,), dtype=int),
            }
        )

        # action_space -> 4 -> left, right, up, down
        self.action_space = spaces.Discrete(4)
        # map direction ->
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # forbidden area
        '''
        The graph should look like this.
        ##### #####
        ##### #####
        ###########
        ##### #####
        ##### #####
          #     #  
        ##### #####
        ##### #####
        ###########
        ##### #####
        ##### #####
        '''
        self.forbidden_area = set()
        self.forbidden_area.update([(size, i) for i in range(0, size * 2 + 1)])
        self.forbidden_area.update([(i, size) for i in range(0, size * 2 + 1)])
        self.forbidden_area.difference_update({(size, size - size // 2 - 1),
                                               (size, size + size // 2 + 1),
                                               (size + size // 2 + 1, size),
                                               (size - size // 2 - 1, size)})

    def __str__(self):
        s = ""
        for i in range(0, self.size * 2 + 1):
            for j in range(0, self.size * 2 + 1):
                if (i, j) in self.forbidden_area:
                    s += ' '
                else:
                    s += '#'
            if i != self.size * 2:
                s += '\n'

        return s

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        # but some of them are not accessible
        while True:
            self._agent_location = self.np_random.integers(
                0, self.size, size=2, dtype=int)
            x, y = self._agent_location
            if not (x, y) in self.forbidden_area:
                break

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while True:
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int)
            x, y = self._target_location
            if not ((x, y) in self.forbidden_area or np.array_equal(self._target_location, self._agent_location)):
                break

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # case1: going to the fobidden area
        case1_flag = False
        tmp_location = self._agent_location + direction
        x, y = tmp_location
        if (x, y) in self.forbidden_area:
            case1_flag = True

        # case2:
        # We use `np.clip` to make sure we don't leave the grid
        if not case1_flag:
            self._agent_location = np.clip(
                tmp_location, 0, self.size * 2
            )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(
            self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # white
        pix_square_size = (
            self.window_size / (self.size * 2 + 1)
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),  # red
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (255, 255, 0),  # yellow
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Then we draw the forbidden area
        for x, y in self.forbidden_area:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),  # black
                pygame.Rect(
                    pix_square_size * np.array([x, y]),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size * 2 + 2):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def main():
    emptyRoom = EmptyRooms(render_mode="human")
    observation, _ = emptyRoom.reset()


if __name__ == '__main__':
    main()
