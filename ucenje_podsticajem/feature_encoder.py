import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder

from ucenje_podsticajem.action_types import ActionTypes
from ucenje_podsticajem.grid_environment import GridEnvironment


def build_dataset(grid_environment, num_episodes):
    enc = OneHotEncoder()
    action_encodings = enc.fit_transform(np.arange(len(ActionTypes)).reshape((len(ActionTypes), 1))).toarray()

    action_list = [ActionTypes(i) for i in range(len(ActionTypes))]

    x = []
    y = []

    for episode in range(num_episodes):
        grid_environment.reset(random.choice(grid_environment.valid_states))
        observation, reward, terminate = grid_environment.step()  # initial step

        while not terminate:
            action = random.choice(action_list)
            x.append(np.concatenate((observation.toArray().squeeze(), action_encodings[action])))

            observation, reward, terminate = grid_environment.step(action)
            y.append(observation.toArray().squeeze())

    return x, y


def train_feature_encoder(num_episodes, grid_environment, **kwargs):
    X, y = build_dataset(grid_environment, num_episodes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=785)

    reg = MLPRegressor(random_state=846515, **kwargs)
    reg.fit(X_train, y_train)

    return reg


if __name__ == '__main__':
    num_episodes = 10000
    grid_environment = GridEnvironment()

    X, y = build_dataset(grid_environment, num_episodes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=785)

    reg = MLPRegressor(random_state=846515)
    reg.fit(X_train, y_train)

    print('Train score: {:2.2f}'.format(reg.score(X_train, y_train)))

    print('Test score: {:2.2f}'.format(reg.score(X_test, y_test)))

    test_samples_num = 10
    y_pred = reg.predict(X_test[:test_samples_num])

    print("Prediction:\n", np.round(y_pred, 0))
    print("True:\n", np.array((y_test[:test_samples_num])))

    weights = reg.coefs_
    biases = reg.intercepts_
    debug_var = None
