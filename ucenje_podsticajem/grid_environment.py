import random

from enum import IntEnum

from ucenje_podsticajem.agent import ActionTypes


class FieldTypes(IntEnum):
    WALL = 0
    NEUTRAL = 1
    TERMINAL = 2


class State2D:
    def __init__(self, row=0, column=0):
        """
        Init a state that consists of (row, column).
        :param row: int
        :param column: int
        :param row_limits: tuple(int, int)
        :param column_limits: tuple(int, int)
        """
        self.row = row
        self.column = column

    def __eq__(self, other):
        if isinstance(other, State2D):
            return (self.row, self.column) == (other.row, other.column)

        return NotImplemented

    def __key(self):
        return self.row, self.column

    def __hash__(self):
        return hash(self.__key())

    def isEqualTo(self, row, column):
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

        self.applied_action = None
        self.current_state = None
        self.env_map = None
        self.height = None
        self.reward_map = None
        self.start_state = None
        self.stochasticity = stochasticity
        self.valid_states = []
        self.width = None

        self.loadEnvMap(env_map)
        self.loadRewardMap(reward_map)

        self.reset(start_state)

    def loadEnvMap(self, env_map):
        """
        Process the given environment map. If it's None load the default one. Assert that the dimensions are valid and load get the list of valid states.
        :param env_map: List[List[FieldTypes]]
        :return:
        """
        if env_map is None:  # default map
            env_map = [
                [FieldTypes.WALL] * 7,
                [FieldTypes.WALL] + [FieldTypes.NEUTRAL] * 5 + [FieldTypes.WALL],
                [FieldTypes.WALL, FieldTypes.TERMINAL] * 3 + [FieldTypes.WALL],
                [FieldTypes.WALL] * 7
            ]

        GridEnvironment.assertRectangular(env_map)
        self.env_map = env_map

        self.height = len(self.env_map)
        self.width = len(self.env_map[0])

        self.getValidStates()

    def loadRewardMap(self, reward_map):
        """
        Process the given reward map. If it's None load the default one. Assert that the dimensions are valid.
        :param reward_map: List[List[int/float]]
        :return:
        """
        if reward_map is None:
            reward_map = [
                [0] * 7,
                [0] * 7,
                [0, -1, 0, -1, 0, 3, 0],
                [0] * 7
            ]

        self.reward_map = reward_map

        GridEnvironment.assertRectangular(self.reward_map)

        # assert that the reward map matches the grid
        assert self.height == len(self.reward_map)
        assert self.width == len(self.reward_map[0])

    def reset(self, start_state=None):
        """
        Reset the environment to the start state.
        :param start_state: State2D
        :return:
        """
        if start_state is None:
            start_state = State2D(1, 1)

        assert start_state in self.valid_states
        self.start_state = start_state

        self.current_state = self.start_state

    def step(self, action=None):
        """
        Pass the given action through the stochastic environment and apply it. Return the current state and reward.
        :param action: ActionTypes
        :return: tuple(State2D, int); current state and reward
        """

        if action is not None:
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

        new_state = State2D(row, column)

        if new_state in self.valid_states:
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
                if self.current_state.isEqualTo(row, column):
                    output = "*" + self.env_map[row][column].name  # denote the current state with a '*'
                else:
                    output = self.env_map[row][column].name

                if self.env_map[row][column] != FieldTypes.WALL:  # add reward to the parts of the map that aren't wall
                    output += "({:d})".format(self.getReward(State2D(row, column)))

                print("{:{cell_size}s}".format(output, cell_size=cell_size), end='')

            print()
        print()

    def getValidStates(self):
        """
        Get all the states that aren't walls.
        :return:
        """
        self.valid_states = []

        for row in range(self.height):
            for column in range(self.width):
                if self.env_map[row][column] is not FieldTypes.WALL:
                    self.valid_states.append(State2D(row=row, column=column))

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

    actions = [ActionTypes.RIGHT, ActionTypes.DOWN]
    for i in range(len(actions)):
        # grid_environment.applyAction(actions[i])
        grid_environment.step(actions[i])
        print(actions[i].name)
        grid_environment.printMap(i + 1)
