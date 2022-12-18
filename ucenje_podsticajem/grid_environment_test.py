import unittest

from collections import Counter

from ucenje_podsticajem.actor import ActionTypes
from ucenje_podsticajem.grid_environment import GridEnvironment


class StochasticityTest(unittest.TestCase):
    def test_distributions(self):
        """
        Test if the stochastic environment was programmed correctly.
        :return:
        """
        stochasticity = 0.4  # probability to change action
        horizontal = [ActionTypes.LEFT, ActionTypes.RIGHT]
        vertical = [ActionTypes.UP, ActionTypes.DOWN]

        num_actions = 10000
        grid_environment = GridEnvironment(stochasticity=stochasticity)

        for direction in [horizontal, vertical]:
            for action in direction:
                action_list = [action] * num_actions
                applied_actions = []
                for i in range(num_actions):  # apply the same action many times
                    grid_environment.step(action_list[i])
                    applied_actions.append(grid_environment.applied_action)

                if direction == horizontal:
                    orthogonal = vertical
                else:
                    orthogonal = horizontal

                counter = Counter(applied_actions)
                try:  # assert that the expected counts are +- 10% good
                    self.assertTrue(int(num_actions * (1 - stochasticity) * 0.9) <= counter[action] <= int(num_actions * (1 - stochasticity) * 1.1))
                    self.assertTrue(int(num_actions * 0.5 * stochasticity * 0.9) <= counter[orthogonal[0]] <= int(num_actions * 0.5 * stochasticity * 1.1))
                    self.assertTrue(int(num_actions * 0.5 * stochasticity * 0.9) <= counter[orthogonal[1]] <= int(num_actions * 0.5 * stochasticity * 1.1))
                except AssertionError:
                    print(counter)
                    raise AssertionError


if __name__ == '__main__':
    unittest.main()
