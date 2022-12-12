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
    def __init__(self, env_map=None, start_state=None):
        """
        Initialize the grid RL environment.
        :param env_map: List[List[FieldTypes]]
        :param start_state: State2D
        """
        self.current_state = None
        self.start_state = None

        if env_map is None:  # default map
            self.env_map = [
                [FieldTypes.WALL] * 7,
                [FieldTypes.WALL] + [FieldTypes.NEUTRAL] * 5 + [FieldTypes.WALL],
                [FieldTypes.WALL, FieldTypes.TERMINAL] * 3 + [FieldTypes.WALL],
                [FieldTypes.WALL] * 7
            ]

        # assert that the grid is rectangular
        row_widths = [len(row) for row in self.env_map]
        assert all([row_widths[0] == row_width for row_width in row_widths]), "The grid is not rectangular. The rows widths are: {:s}".format(str(row_widths))

        self.height = len(self.env_map)
        self.width = len(self.env_map[0])
        self.row_limits = (1, self.height - 2)
        self.column_limits = (1, self.width - 2)

        self.reset(start_state)

    def reset(self, start_state=None):
        """
        Reset the environment to the start state
        :param start_state: State2D
        :return:
        """
        if start_state is None:
            self.start_state = State2D(1, 1, self.row_limits, self.column_limits)

        assert self.start_state.inLimits()  # assert that the state is in bounds

        self.current_state = self.start_state

    def step(self, action):
        # TODO
        #  stochastically change the action
        #  apply action
        #  return the reward and observation(current state)
        raise NotImplemented

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

    def printMap(self, step=0, action=None, cell_size=10):
        """
        Print the map and the current state denoted by a '*'.
        :return:
        """

        # print delimiter and timestep
        print("{:s} {:s}".format("-" * cell_size * (len(self.env_map[0]) + 1),
                                 "step = {:d}, ".format(step)), end='')

        # action taken
        if hasattr(action, 'name'):
            print("action = {:s}, ".format(action.name), end='')

        print("state = ({:<d}, {:<d})".format(self.current_state.row, self.current_state.column))

        print("{:{cell_size}s}".format("", cell_size=cell_size), end='')
        for i in range(len(self.env_map[0])):
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

                print("{:{cell_size}s}".format(output, cell_size=cell_size), end='')

            print()


if __name__ == '__main__':
    grid_environment = GridEnvironment()
    grid_environment.printMap()

    actions = [ActionTypes.RIGHT, ActionTypes.UP, ActionTypes.RIGHT, ActionTypes.DOWN, ActionTypes.UP] + [ActionTypes.RIGHT] * 3 + [ActionTypes.LEFT] * 8 + [ActionTypes.DOWN] * 3
    for i in range(len(actions)):
        grid_environment.applyAction(actions[i])
        grid_environment.printMap(i + 1, actions[i])
