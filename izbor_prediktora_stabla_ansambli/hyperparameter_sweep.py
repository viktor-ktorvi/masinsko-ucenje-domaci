import argparse
import os
import wandb
import yaml

import numpy as np

from sklearn.metrics import accuracy_score

from data_loading.data_loading import load_data
from izbor_prediktora_stabla_ansambli import random_forest, gradient_boosting
from utils.validation import repeated_k_fold


def train(x, y, config_path, train_and_predict_foo, n_splits=5, n_repeats=2):
    """

    :param x: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y: np.ndarray; shape num_samples x 1; dataset labels
    :param config_path: string; wandb condig defaults path
    :param n_splits: int; k in k-fold validation
    :param n_repeats: int; number of times to do k-fold validation
    :return:
    """
    wandb.init(config=config_path)
    kwargs = {'n_estimators': wandb.config.n_estimators,
              'max_depth': wandb.config.max_depth,
              'random_state': wandb.config.random_state}

    if hasattr(wandb.config, 'learning_rate'):
        kwargs['learning_rate'] = wandb.config.learning_rate

    if hasattr(wandb.config, 'max_features'):
        kwargs['max_features'] = wandb.config.max_features

    accuracies = repeated_k_fold(x, y,
                                 train_and_predict_foo=lambda x_train, y_train, x_test: train_and_predict_foo(x_train, y_train, x_test, **kwargs),
                                 metric_foo=accuracy_score, n_splits=n_splits, n_repeats=n_repeats, random_state=wandb.config.random_state)

    wandb.log({'accuracy': np.mean(accuracies)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, nargs='?', default=None)
    parser.add_argument('--classifier_name', type=str, required=True)

    args = parser.parse_args()
    print("Passed Arguments:\n", args)

    if args.classifier_name.lower() == 'random_forest':
        sweep_config_name = 'random-forest-sweep-config.yaml'
        defaults_config_name = 'random-forest-config-defaults.yaml'

    elif args.classifier_name.lower() == 'gradient_boosting':
        sweep_config_name = 'gradient-boosting-sweep-config.yaml'
        defaults_config_name = 'gradient-boosting-config-defaults.yaml'

    else:
        raise ValueError("Classifier '{:s}' is not supported".format(args.classifier_name.lower()))

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    sweep_config_path = os.path.join(__location__, sweep_config_name)

    with open(sweep_config_path, 'r') as stream:
        try:
            sweep_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    csv_path = os.path.join(__location__, "data_2.csv")
    x, y = load_data(csv_path)


    def train_random_forest():
        train(x, y, config_path=os.path.join(__location__, defaults_config_name), train_and_predict_foo=random_forest.train_and_predict_foo, n_splits=5, n_repeats=2)


    def train_gradient_boost():
        train(x, y, config_path=os.path.join(__location__, defaults_config_name), train_and_predict_foo=gradient_boosting.train_and_predict_foo, n_splits=5, n_repeats=2)


    train_foo = train_random_forest if args.classifier_name.lower() == 'random_forest' else train_gradient_boost

    sweep_id = wandb.sweep(sweep_config, project="mu-domaci-random-forest-and-gradient-boosting")
    wandb.agent(sweep_id, function=train_foo, count=args.count)
