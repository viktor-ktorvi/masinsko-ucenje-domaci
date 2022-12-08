import os
import wandb

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils.validation import repeated_k_fold

from data_loading.data_loading import load_data


def train_and_predict_foo(x_train, y_train, x_test, n_estimators, max_depth, max_features, random_state):
    """

    :param x_train: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y_train: np.ndarray; shape num_samples x 1; dataset labels
    :param x_test: np.ndarray; shape num_test_samples x num_features; test dataset feature matrix
    :param n_estimators: int; number of models in the ansamble
    :param max_depth: int; maximum depth of decision trees
    :param max_features: int; number of features to consider when looking for the best split
    :param random_state: int; random seed for reproducibility purposes
    :return: np.ndarray; shape num_test_samples x 1; test predictions
    """
    random_forest_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=random_state)
    random_forest_clf.fit(x_train, y_train.squeeze())

    return random_forest_clf.predict(x_test)


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path_1 = os.path.join(__location__, 'data_2.csv')  # data path

    wandb.init(project="mu-domaci-random-forest", config=os.path.join(__location__, 'random-forest-config-defaults.yaml'), mode='disabled')

    x, y = load_data(csv_path_1)  # load data

    accuracies = repeated_k_fold(x, y,
                                 train_and_predict_foo=lambda x_train, y_train, x_test: train_and_predict_foo(x_train, y_train, x_test,
                                                                                                              n_estimators=wandb.config.n_estimators,
                                                                                                              max_depth=wandb.config.max_depth,
                                                                                                              max_features=wandb.config.max_features,
                                                                                                              random_state=wandb.config.random_state),
                                 metric_foo=accuracy_score, n_splits=5, n_repeats=2, random_state=wandb.config.random_state)

    print(pd.DataFrame(accuracies).describe())
