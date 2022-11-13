import os
import wandb
import yaml

import numpy as np

from sklearn.metrics import accuracy_score

from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.model import train_one_vs_rest
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.logistic_regression import \
    predict_multiclass

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

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    default_config_path = os.path.join(__location__, "config-defaults.yaml")

    with wandb.init(project="mu-domaci-logisticka-regresija", config=default_config_path):
        classifiers, loggers = train_one_vs_rest(X_train_transformed, y_train, epochs=wandb.config.epochs,
                                                 lr=wandb.config.lr, batch_size=wandb.config.batch_size,
                                                 log=True)

        test_predictions = predict_multiclass(X_test_transformed, classifiers)

        x_values = np.arange(wandb.config.epochs) + 1

        wandb.log(
            {
                # "loss": wandb.plot.line_series(
                #     xs=x_values,
                #     ys=[loggers[i].loss for i in range(len(loggers))],
                #     keys=["Klasa {:d}".format(i) for i in range(len(loggers))],
                #     title="Apsolutna greska",
                #     xname="epohe"),
                # "accuracy": wandb.plot.line_series(
                #     xs=x_values,
                #     ys=[loggers[i].accuracy for i in range(len(loggers))],
                #     keys=["Klasa {:d}".format(i) for i in range(len(loggers))],
                #     title="Tacnost",
                #     xname="epohe"),
                "test accuracy": accuracy_score(y_test, test_predictions)
            })


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    sweep_config_path = os.path.join(__location__, "hyperparameter_sweep.yaml")

    with open(sweep_config_path, 'r') as stream:
        try:
            sweep_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_path = os.path.join(__location__, "../multiclass_data.csv")
    X_train_transformed, X_test_transformed, y_train, y_test = load_transformed_data(data_path,
                                                                                     test_size=0.2, norm=True,
                                                                                     dimensionality=None)

    sweep_id = wandb.sweep(sweep_config, project="mu-domaci-logisticka-regresija")
    wandb.agent(sweep_id, function=lambda: logged_training(X_train_transformed, X_test_transformed, y_train, y_test),
                count=200)
