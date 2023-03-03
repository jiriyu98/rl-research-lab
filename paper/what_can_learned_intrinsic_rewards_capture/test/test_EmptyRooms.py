import unittest
import numpy as np

from paper.what_can_learned_intrinsic_rewards_capture.envs.EmptyRooms import EmptyRooms


class TestEmptyRooms(unittest.TestCase):
    def test_emptyRooms_init(self):
        emptyRooms = EmptyRooms()
        # https://www.pythonescaper.com/
        expected_str = ("##### #####\n"
                        "##### #####\n"
                        "###########\n"
                        "##### #####\n"
                        "##### #####\n"
                        "  #     #  \n"
                        "##### #####\n"
                        "##### #####\n"
                        "###########\n"
                        "##### #####\n"
                        "##### #####")
        self.assertEqual(emptyRooms.__str__(), expected_str)

    def test_reset(self):
        emptyRooms = EmptyRooms()

        observation, info = emptyRooms.reset(seed=0)
        # when seed=0, agent_location=(4, 3), target_location=(2, 1)
        expected_agent_location = np.array([4, 3])
        expected_target_location = np.array([2, 1])

        np.testing.assert_equal(observation["agent"],
                                expected_agent_location)
        np.testing.assert_equal(observation["target"],
                                expected_target_location)

    def test_step(self):
        emptyRooms = EmptyRooms()

        observation, info = emptyRooms.reset(seed=0)
        # when seed=0, agent_location=(4, 3), target_location=(2, 1)
        # _action_to_direction = {
        #     0: np.array([1, 0]),
        #     1: np.array([0, 1]),
        #     2: np.array([-1, 0]),
        #     3: np.array([0, -1]),
        # }

        observation, _, _, _, _ = emptyRooms.step(0)
        np.testing.assert_equal(observation["agent"],
                                np.array([4, 3]))
        observation, _, _, _, _ = emptyRooms.step(3)
        np.testing.assert_equal(observation["agent"],
                                np.array([4, 2]))
        observation, _, _, _, _ = emptyRooms.step(0)
        np.testing.assert_equal(observation["agent"],
                                np.array([5, 2]))
        observation, _, _, _, _ = emptyRooms.step(3)
        np.testing.assert_equal(observation["agent"],
                                np.array([5, 2]))
        observation, _, _, _, _ = emptyRooms.step(1)
        np.testing.assert_equal(observation["agent"],
                                np.array([5, 2]))
        observation, _, _, _, _ = emptyRooms.step(0)
        np.testing.assert_equal(observation["agent"],
                                np.array([6, 2]))
        observation, _, _, _, _ = emptyRooms.step(3)
        np.testing.assert_equal(observation["agent"],
                                np.array([6, 1]))
        observation, _, _, _, _ = emptyRooms.step(3)
        np.testing.assert_equal(observation["agent"],
                                np.array([6, 0]))
        observation, _, _, _, _ = emptyRooms.step(3)
        np.testing.assert_equal(observation["agent"],
                                np.array([6, 0]))


if __name__ == '__main__':
    unittest.main()
