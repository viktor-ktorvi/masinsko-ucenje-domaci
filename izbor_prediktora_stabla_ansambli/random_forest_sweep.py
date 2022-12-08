import argparse
import os
import wandb
import yaml

import numpy as np

from sklearn.metrics import accuracy_score

from data_loading.data_loading import load_data
from izbor_prediktora_stabla_ansambli.random_forest import train_and_predict_foo
from utils.validation import repeated_k_fold


def train(x, y, config_path, n_splits=5, n_repeats=2):
    """

    :param x: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y: np.ndarray; shape num_samples x 1; dataset labels
    :param config_path: string; wandb condig defaults path
    :param n_splits: int; k in k-fold validation
    :param n_repeats: int; number of times to do k-fold validation
    :return:
    """
    wandb.init(config=config_path)

    accuracies = repeated_k_fold(x, y,
                                 train_and_predict_foo=lambda x_train, y_train, x_test: train_and_predict_foo(x_train, y_train, x_test,
                                                                                                              n_estimators=wandb.config.n_estimators,
                                                                                                              max_depth=wandb.config.max_depth,
                                                                                                              max_features=wandb.config.max_features,
                                                                                                              random_state=wandb.config.random_state),
                                 metric_foo=accuracy_score, n_splits=n_splits, n_repeats=n_repeats, random_state=wandb.config.random_state)

    wandb.log({'accuracy': np.mean(accuracies)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, nargs='?', default=None)
    args = parser.parse_args()
    print("Passed Arguments:\n", args)

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    sweep_config_path = os.path.join(__location__, "random-forest-sweep-config.yaml")

    with open(sweep_config_path, 'r') as stream:
        try:
            sweep_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    csv_path = os.path.join(__location__, "data_2.csv")
    x, y = load_data(csv_path)

    sweep_id = wandb.sweep(sweep_config, project="mu-domaci-random-forest")
    wandb.agent(sweep_id, function=lambda: train(x, y, config_path=os.path.join(__location__, 'random-forest-config-defaults.yaml'), n_splits=5, n_repeats=2), count=args.count)
