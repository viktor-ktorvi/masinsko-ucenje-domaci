from enum import IntEnum


class FieldTypes(IntEnum):
    WALL = 0
    NEUTRAL = 1
    TERMINAL = 2


class State2D:
    def __init__(self, row=0, column=0, row_limits=None, column_limits=None):
        self.row = row
        self.column = column
        self.row_limits = row_limits
        self.column_limits = column_limits

    def inLimits(self):
        return self.row_limits[0] <= self.row <= self.row_limits[1] and self.column_limits[0] <= self.column <= self.column_limits[1]

    def isEqual(self, row, column):
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

        if env_map is None:
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

        self.reset(start_state)

    def reset(self, start_state=None):
        """
        Reset the environment to the start state
        :param start_state: State2D
        :return:
        """
        if start_state is None:
            self.start_state = State2D(1, 1, (1, self.height - 1), (1, self.width - 1))

        assert self.start_state.inLimits()  # assert that the state is in bounds

        self.current_state = self.start_state

    def step(self, action):
        raise NotImplemented

    def printMap(self, step=0, cell_size=10):
        """
        Print the map and the current state denoted by a '*'.
        :return:
        """
        print("-" * cell_size * (len(self.env_map[0]) + 1), end=' step = ' + str(step) + '\n')
        print("{:{cell_size}s}".format("", cell_size=cell_size), end='')
        for i in range(len(self.env_map[0])):
            print("{:{cell_size}s}".format(str(i), cell_size=cell_size), end='')
        print()

        for row in range(len(self.env_map)):
            print("{:{cell_size}s}".format(str(row), cell_size=cell_size), end='')
            for column in range(len(self.env_map[0])):

                if self.current_state.isEqual(row, column):
                    output = "*" + self.env_map[row][column].name
                else:
                    output = self.env_map[row][column].name

                print("{:{cell_size}s}".format(output, cell_size=cell_size), end='')

            print()


if __name__ == '__main__':
    grid_environment = GridEnvironment()
    grid_environment.printMap()
