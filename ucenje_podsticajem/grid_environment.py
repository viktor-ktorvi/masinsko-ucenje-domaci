import random

from enum import IntEnum

from ucenje_podsticajem.actor import ActionTypes


class FieldTypes(IntEnum):
    WALL = 0
    NEUTRAL = 1
    TERMINAL = 2


class State2D:
    def __init__(self, row=0, column=0, row_limits=None, column_limits=None):
        """
        Init a state that consists of (row, column).
        :param row: int
        :param column: int
        :param row_limits: tuple(int, int)
        :param column_limits: tuple(int, int)
        """
        self.row = row
        self.column = column
        self.row_limits = row_limits
        self.column_limits = column_limits

    def inLimits(self):
        """
        Check if the state is within the limits.
        :return: boolean
        """
        return self.row_limits[0] <= self.row <= self.row_limits[1] and self.column_limits[0] <= self.column <= self.column_limits[1]

    def isEqual(self, row, column):
        """
        Check if the state is equal to a row column pair.
        :param row: int
        :param column: int
        :return: boolean
        """
        return row == self.row and column == self.column


class GridEnvironment:
    def __init__(self, env_map=None, reward_map=None, start_state=None, stochasticity=0.4):
        """
        Initialize the grid RL environment.
        :param env_map: List[List[FieldTypes]]
        :param start_state: State2D
        """
        self.stochasticity = stochasticity
        self.current_state = None
        self.start_state = None

        if env_map is None:  # default map
            env_map = [
                [FieldTypes.WALL] * 7,
                [FieldTypes.WALL] + [FieldTypes.NEUTRAL] * 5 + [FieldTypes.WALL],
                [FieldTypes.WALL, FieldTypes.TERMINAL] * 3 + [FieldTypes.WALL],
                [FieldTypes.WALL] * 7
            ]

        self.env_map = env_map

        if reward_map is None:
            reward_map = [
                [0] * 7,
                [0] * 7,
                [0, -1, 0, -1, 0, 3, 0],
                [0] * 7
            ]

        self.reward_map = reward_map

        # assert that the grid is rectangular
        GridEnvironment.assertRectangular(self.env_map)
        GridEnvironment.assertRectangular(self.reward_map)

        self.height = len(self.env_map)
        self.width = len(self.env_map[0])
        self.row_limits = (1, self.height - 2)
        self.column_limits = (1, self.width - 2)

        # assert that the reward map matches the grid
        assert self.height == len(self.reward_map)
        assert self.width == len(self.reward_map[0])

        self.applied_action = None

        self.reset(start_state)

    def reset(self, start_state=None):
        """
        Reset the environment to the start state.
        :param start_state: State2D
        :return:
        """
        if start_state is None:
            self.start_state = State2D(1, 1, self.row_limits, self.column_limits)

        assert self.start_state.inLimits()  # assert that the state is in bounds

        self.current_state = self.start_state

    def step(self, action):
        """
        Pass the given action through the stochastic environment and apply it. Return the current state and reward.
        :param action: ActionTypes
        :return: tuple(State2D, int); current state and reward
        """
        # stochastically modify the action
        coin_toss = random.choices([True, False], weights=[self.stochasticity, 1 - self.stochasticity])[0]
        if coin_toss:
            self.applied_action = GridEnvironment.randomOrthogonalAction(action)
        else:
            self.applied_action = action

        # apply the action
        self.applyAction(self.applied_action)

        terminal_flag = self.env_map[self.current_state.row][self.current_state.column] == FieldTypes.TERMINAL

        # return the observation and reward
        return self.current_state, self.getReward(self.current_state), terminal_flag

    def applyAction(self, action):
        """
        Apply the action to the environment and update the current state.
        :param action: ActionTypes
        :return:
        """
        row, column = self.current_state.row, self.current_state.column

        if action == ActionTypes.UP:
            row -= 1
        elif action == ActionTypes.DOWN:
            row += 1
        elif action == ActionTypes.LEFT:
            column -= 1
        elif action == ActionTypes.RIGHT:
            column += 1
        else:
            raise ValueError("Action '{:s}' not supported.".format(action))

        new_state = State2D(row, column, self.row_limits, self.column_limits)

        if new_state.inLimits():  # if the new state is within limits then update the current state
            self.current_state = new_state

    def getReward(self, state):
        """
        Get the reward of the given state.
        :param state: Stade2D
        :return: int; reward value
        """
        row = state.row
        column = state.column
        return self.reward_map[row][column]

    def printMap(self, step=0, cell_size=15):
        """
        Print the map and the current state denoted by a '*'.
        :param step: int; time step
        :param cell_size: int;
        :return:
        """

        # print delimiter and timestep
        print("{:s} {:s}".format("-" * cell_size * (len(self.env_map[0]) + 1),
                                 "step = {:d}, ".format(step)), end='')

        # action taken
        if hasattr(self.applied_action, 'name'):
            print("action = {:s}, ".format(self.applied_action.name), end='')

        print("state = ({:<d}, {:<d})".format(self.current_state.row, self.current_state.column))

        print("{:{cell_size}s}".format("", cell_size=cell_size), end='')  # empty cell

        for i in range(len(self.env_map[0])):  # column ids
            print("{:{cell_size}s}".format(str(i), cell_size=cell_size), end='')
        print()

        # print map
        for row in range(len(self.env_map)):
            print("{:{cell_size}s}".format(str(row), cell_size=cell_size), end='')
            for column in range(len(self.env_map[0])):
                if self.current_state.isEqual(row, column):
                    output = "*" + self.env_map[row][column].name  # denote the current state with a '*'
                else:
                    output = self.env_map[row][column].name

                if self.env_map[row][column] != FieldTypes.WALL:  # add reward to the parts of the map that aren't wall
                    output += "({:d})".format(self.getReward(State2D(row, column)))

                print("{:{cell_size}s}".format(output, cell_size=cell_size), end='')

            print()
        print()

    @staticmethod
    def assertRectangular(double_list):
        """
        Assert that the double list is rectangular i.e. every row has the same width.
        :param double_list: List[List[object]]
        :return:
        """
        row_widths = [len(row) for row in double_list]
        assert all([row_widths[0] == row_width for row_width in row_widths]), "The list is not rectangular. The rows widths are: {:s}".format(str(row_widths))

    @staticmethod
    def randomOrthogonalAction(action):
        """
        Return a random action orthogonal to the given action.
        :param action: ActionTypes
        :return: ActionTypes
        """
        horizontal = [ActionTypes.LEFT, ActionTypes.RIGHT]
        vertical = [ActionTypes.UP, ActionTypes.DOWN]

        if action in horizontal:
            return random.choice(vertical)

        if action in vertical:
            return random.choice(horizontal)

        raise ValueError("Action '{action}' is not supported.".format(action=action))


if __name__ == '__main__':
    grid_environment = GridEnvironment()
    grid_environment.printMap()

    actions = [ActionTypes.RIGHT, ActionTypes.UP, ActionTypes.RIGHT, ActionTypes.DOWN, ActionTypes.UP] + [ActionTypes.RIGHT] * 3 + [ActionTypes.LEFT] * 8 + [ActionTypes.DOWN] * 3
    for i in range(len(actions)):
        # grid_environment.applyAction(actions[i])
        grid_environment.step(actions[i])
        print(actions[i].name)
        grid_environment.printMap(i + 1)
