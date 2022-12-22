import random

import numpy as np

from enum import IntEnum
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder


class ActionTypes(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class AgentQ:
    def __init__(self, state_height, state_width, init_learning_rate=1, init_epsilon=0.999, gamma=0.9):
        """
        Initialize the Q learning agent.
        :param state_height: int
        :param state_width: int
        :param init_learning_rate: float
        :param init_epsilon: float; greedy exploration parameter
        :param gamma: float; discount factor
        """
        self.Q_table = None
        self.epsilon = None
        self.gamma = gamma
        self.init_epsilon = init_epsilon
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


class AgentREINFORCE:
    def __init__(self, state_dimension):
        self.theta = np.random.randn(state_dimension + len(ActionTypes), 1)

        enc = OneHotEncoder()
        self.action_encodings = enc.fit_transform(np.arange(len(ActionTypes)).reshape((len(ActionTypes), 1))).toarray()

        self.action_list = [ActionTypes(i) for i in range(len(ActionTypes))]

    def policy(self, state):
        """
        Calculate the agents' policy for the given state. Return the features as well.
        :param state: State2D
        :return: tuple(np.ndarray); np.ndarray; shape num_actions x 1 - policy |
                                    np.ndarray; shape num_actions x (state_dimension + num_actions) features
        """
        features = self.constructFeatures(state)
        policy = softmax(features @ self.theta)  # calculate the policy

        return policy, features

    def act(self, state):
        """
        Calculate the agents policy and sample an action from it.
        :param state: State2D
        :return: ActionTypes
        """
        policy, _ = self.policy(state)

        action = random.choices(self.action_list, weights=list(policy.flatten()))[0]  # sample from the policy

        return action

    def score(self, state):
        policy, features = self.policy(state)

        score = features - np.sum(features * policy, axis=0)

        return score

    def observe(self, observations, actions, rewards, learning_rate, gamma):
        assert len(observations) == len(actions)
        assert len(actions) == len(rewards)

        v = np.correlate(gamma ** np.arange(len(rewards)), rewards, "full")[:len(rewards)]

        scores = []
        for i in range(len(observations)):
            scores.append(self.score(observations[i])[actions[i]].reshape(self.theta.shape))    # num_time_steps x num_features x 1

        # multiply scores and values timestep wise and sum them up
        total_update = np.sum(v[:, np.newaxis, np.newaxis] * np.array(scores), axis=0)

        self.theta += learning_rate * total_update

    def constructFeatures(self, state):
        """
        concatenate the state and action encodings
        :param state: State2d
        :return: np.ndarray; shape num_actions x (state_dimension + num_actions)
        """
        return np.hstack((np.tile(state.toArray(), (len(ActionTypes), 1)), self.action_encodings))


if __name__ == '__main__':
    agentQ = AgentQ(4, 7)
    print(agentQ.Q_table.shape)

    agentR = AgentREINFORCE(2)

    from ucenje_podsticajem.grid_environment import State2D

    state = State2D(1, 3)
    agentR.act(state)

    debug_var = None
