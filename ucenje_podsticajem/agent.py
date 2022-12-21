import numpy as np

from enum import IntEnum


class ActionTypes(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class AgentQ:
    def __init__(self, state_height, state_width, init_learning_rate=1, epsilon=0.999, gamma=0.9):
        """
        Initialize the Q learning agent.
        :param state_height: int
        :param state_width: int
        :param init_learning_rate: float
        :param epsilon: float; greedy exploration parameter
        :param gamma: float; discount factor
        """
        self.Q_table = None
        self.epsilon = None
        self.gamma = gamma
        self.init_epsilon = epsilon
        self.init_learning_rate = init_learning_rate
        self.prev_action = None
        self.prev_state = None
        self.state_height = state_height
        self.state_width = state_width
        self.train_mode = False

        self.fullReset()

    def fullReset(self):
        """
        Reset everything in the agent.
        :return:
        """

        self.epsilon = self.init_epsilon
        self.Q_table = np.zeros((self.state_height, self.state_width, len(ActionTypes)), dtype=np.float32)
        self.memoryReset()

    def memoryReset(self):
        """
        Reset the memory of the previous state and action.
        :return:
        """
        self.prev_action = None
        self.prev_state = None

    def train(self):
        """
        Set the agent to training mode.
        :return:
        """
        self.train_mode = True

    def eval(self):
        """
        Set the agent to evaluation mode.
        :return:
        """
        self.train_mode = False

    def observe(self, state, reward, learning_rate):
        """
        Update the Q table if the agent is in the training mode.
        :param state: State2D
        :param reward: int/float
        :param epoch: timestep
        :return:
        """
        if self.train_mode and self.prev_state is not None:
            self.update(state, reward, learning_rate)

        self.prev_state = state

    def update(self, state, reward, learning_rate):
        """
        Update Q table using temporal difference learning.
        :param state: State2D
        :param reward: int/float
        :param learning_rate: float
        :return:
        """
        q_estimate = reward + self.gamma * np.max(self.Q_table[state.row, state.column, :])  # action value function estimate

        temporal_difference = q_estimate - self.Q_table[self.prev_state.row, self.prev_state.column, self.prev_action]  # temporal difference of estimate VS observed action value function

        self.Q_table[self.prev_state.row, self.prev_state.column, self.prev_action] += learning_rate * temporal_difference  # update

    def getMaxQ(self, state):
        """
        Return the maximum value of the action value function for the given state.
        :param state: State2D
        :return: float
        """
        return np.max(self.Q_table[state.row, state.column, :])

    def act(self, state):
        """
        Choose the best action for the given state.
        :param state: State2D
        :return: ActionTypes
        """

        if self.train_mode and np.random.rand() < self.epsilon:  # choose random action at random times
            self.epsilon *= self.init_epsilon
            action = ActionTypes(np.random.randint(low=0, high=len(ActionTypes)))
        else:
            action = ActionTypes(np.argmax(self.Q_table[state.row, state.column, :]))  # choose optimal action

        self.prev_action = action

        return action


if __name__ == '__main__':
    agent = AgentQ(4, 7)
    print(agent.Q_table.shape)
