import wandb
import yaml

import numpy as np

from sklearn.metrics import accuracy_score

from model import train_one_vs_rest
from logistic_regression import predict_multiclass

from data_loading.data_loading import load_transformed_data


def logged_training(X_train_transformed, X_test_transformed, y_train, y_test):
    """
    Train, test and log.
    :param X_train_transformed: np.ndarray; shape num_train_samples x num_features
    :param X_test_transformed: np.ndarray; shape num_test_samples x num_features
    :param y_train: np.ndarray; shape num_train_samples x 1
    :param y_test: np.ndarray; shape num_test_samples x 1
    :return:
    """
    with wandb.init(project="mu-domaci-logisticka-regresija"):
        classifiers, loggers = train_one_vs_rest(X_train_transformed, y_train, epochs=wandb.config.epochs,
                                                 lr=wandb.config.lr, batch_size=wandb.config.batch_size,
                                                 log=True)

        test_predictions = predict_multiclass(X_test_transformed, classifiers)

        x_values = np.arange(wandb.config.epochs) + 1

        wandb.log(
            {
                "loss": wandb.plot.line_series(
                    xs=x_values,
                    ys=[loggers[i].loss for i in range(len(loggers))],
                    keys=["Klasa {:d}".format(i) for i in range(len(loggers))],
                    title="Apsolutna greska",
                    xname="epohe"),
                "accuracy": wandb.plot.line_series(
                    xs=x_values,
                    ys=[loggers[i].accuracy for i in range(len(loggers))],
                    keys=["Klasa {:d}".format(i) for i in range(len(loggers))],
                    title="Tacnost",
                    xname="epohe"),
                "test accuracy": accuracy_score(y_test, test_predictions)
            })


if __name__ == '__main__':
    with open("hyperparameter_sweep.yaml", 'r') as stream:
        try:
            sweep_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    X_train_transformed, X_test_transformed, y_train, y_test = load_transformed_data('../multiclass_data.csv',
                                                                                     test_size=0.2, norm=True,
                                                                                     dimensionality=2)

    sweep_id = wandb.sweep(sweep_config, project="mu-domaci-logisticka-regresija")
    wandb.agent(sweep_id, function=lambda: logged_training(X_train_transformed, X_test_transformed, y_train, y_test),
                count=1)
