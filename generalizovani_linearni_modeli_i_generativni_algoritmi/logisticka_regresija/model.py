import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from data_loading.data_loading import load_data, add_bias
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.logistic_regression import \
    logistic_regression
from utils.utils import normalize, DataLoader
from utils.logger import ClassificationLogger
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    loss_and_acc_visualization, one_vs_all_visualization, dataset_area_class_visualization


def one_vs_rest_labels(y):
    """
    Assign binary labels to samples such that one class gets the label 1 and the rest 0. Do this for every class and
    return a list.
    :param y: np.ndarray; shape num_samples x 1; multiclass labels
    :return: List[np.ndarray]; len num_classes
    """
    n = len(np.unique(y))
    labels = []
    for i in range(n):
        y_ = np.zeros(y.shape)
        y_[y == i] = 1
        labels.append(y_)

    return labels


def train(x, y, epochs=1, lr=3e-4, batch_size=1, log=False):
    """
    Train a logistic regression classifier.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param y: np.ndarray; shape num_samples x 1; binary labels
    :param epochs: int; number of times to pass through the dataset
    :param lr: float; learning rate
    :param batch_size: int; minibatch size
    :param log: bool; to log losses and accuracies or not
    :return: np.ndarray; shape num_features x 1; classifier coefficients or a tuple of (classifier, Logger)
    """
    x_biased = add_bias(x)  # concat a column of ones

    num_samples, num_features = x_biased.shape
    init_interval = np.sqrt(6) / np.sqrt(num_features + 1)  # normalized Xavier init sqrt(6) / (fan_in + fan_out)
    theta = np.random.uniform(low=-init_interval, high=init_interval, size=(num_features, 1))  # init weights randomly

    train_loader = DataLoader(num_samples, batch_size)
    logger = ClassificationLogger(epochs)

    for i in tqdm(range(epochs)):
        batches = train_loader.get_batches()  # batch ids

        logger.clear()
        for batch_ids in batches:
            batch_x = x_biased[batch_ids]
            batch_y = y[batch_ids]

            y_pred = logistic_regression(batch_x, theta)  # inference

            errors = batch_y - y_pred  # errors batch_size x 1

            mean_update = np.mean(errors * batch_x, axis=0)  # mean LR gradient over the minibatch

            theta += lr * mean_update.reshape(theta.shape)  # update

            logger.running_update(np.sum(np.abs(errors)), acc_cnt=np.sum(np.round(y_pred) == batch_y))  # log

        logger.update(i, scale_factor=num_samples)  # log

    if log:
        return theta, logger

    return theta


def train_one_vs_rest(x_train, y_train, epochs=1, lr=3e-4, batch_size=1, log=False):
    """
    Train N(=number of classes) 'one-vs-rest' classifiers.
    :param x_train: np.ndarray; shape num_samples x num_features; train feature matrix
    :param y_train: np.ndarray; shape num_samples x 1; multiclass labels
    :param epochs: int; number of times to pass through the dataset
    :param lr: float; learning rate
    :param batch_size: int; minibatch size
    :param log: bool; to log losses and accuracies or not
    :return: List[np.ndarray; shape num_features x 1; classifier coefficients] or tuple(List[np.ndarray; shape
    num_features x 1; classifier coefficients], List[Logger])
    """
    labels = one_vs_rest_labels(y_train)

    classifiers = []
    loggers = []
    for i in range(len(labels)):
        out = train(x_train, labels[i], epochs=epochs, lr=lr, batch_size=batch_size, log=log)

        if log:
            classifiers.append(out[0])
            loggers.append(out[1])
        else:
            classifiers.append(out)

    if log:
        return classifiers, loggers

    return classifiers


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path = os.path.join(__location__, '../multiclass_data.csv')

    x, y = load_data(csv_path)

    # split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=546315)

    # transform
    X_tr_norm, mean, std = normalize(X_train)
    X_test_norm = normalize(X_test, mean, std)

    pca = PCA(n_components=2)  # 2d for visualization purposes
    pca.fit(X_tr_norm)

    X_train_transformed = pca.transform(X_tr_norm)
    X_test_transformed = pca.transform(X_test_norm)

    # one vs rest labels
    train_labels = one_vs_rest_labels(y_train)
    test_labels = one_vs_rest_labels(y_test)

    # train one vs rest classifiers
    classifiers, loggers = train_one_vs_rest(X_train_transformed, y_train, epochs=1000, lr=0.003, batch_size=32,
                                             log=True)

    # visualization
    # dataset_visualization(X_train_transformed, y_train)  # train set with labels

    loss_and_acc_visualization(loggers)  # training graphs

    one_vs_all_visualization(classifiers, X_train_transformed, X_test_transformed, train_labels,
                             test_labels)  # trained classifiers

    x_norm = normalize(x, mean, std)
    x_transform = pca.transform(x_norm)

    dataset_area_class_visualization(x_transform, y, classifiers, resolution=(200, 200))  # class areas

    plt.show()
