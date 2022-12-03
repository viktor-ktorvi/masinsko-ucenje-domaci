import argparse
import os

import numpy as np

from enum import IntEnum

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization

from metoda_nosecih_vektora.dual import SVMDual
from metoda_nosecih_vektora.primal import SVMPrimal

from utils.utils import cross_norm_sqrd
from utils.validation import hyperparameter_search


class SVMSolverType(IntEnum):
    Primal = 1
    Dual = 2


def radial_basis_kernel(x1, x2, sigma):
    """
    Return a radial basis kernel(Gaussian) matrix for the given samples.
    :param x1: np.ndarray; shape N x num_features
    :param x2: np.ndarray; shape M x num_features
    :param sigma: float; standard deviation of the Gaussian function
    :return: np.ndarray; shape M x N; rdb/Gaussian kernel
    """
    return np.exp(-0.5 * cross_norm_sqrd(x1, x2) / sigma)


def add_support_vector_visualization(current_axis, x, y, clf):
    """
    Highlight support vectors, draw margins and annotate slack variable values.

    :param current_axis: axis object
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param y: np.ndarray; shape num_samples x 1; sample labels in {-1, 1}
    :param clf: classifier object
    :return:
    """
    support_vectors = x[clf.sv_bool]
    support_vector_labels = y[clf.sv_bool].squeeze()

    support_vector_projection = clf.project(support_vectors)
    support_vector_slacks = np.zeros((support_vectors.shape[0],))

    nonzero_slack_id = support_vector_labels * support_vector_projection < 1
    support_vector_slacks[nonzero_slack_id] = np.abs(support_vector_labels[nonzero_slack_id] - support_vector_projection[nonzero_slack_id])

    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    y_min = np.min(x[:, 1])
    y_max = np.max(x[:, 1])

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_lim = [-0.1 * x_range + x_min, 0.1 * x_range + x_max]
    y_lim = [-0.1 * y_range + y_min, 0.1 * y_range + y_max]

    x_linspace = np.linspace(*x_lim, 100)

    for clss in np.unique(support_vector_labels):
        current_axis.scatter(*[support_vectors[support_vector_labels == clss, i] for i in range(2)],
                             facecolor='none',
                             edgecolor='lime',
                             linewidths=2,
                             label='noseći vektori')

        if type(clf) == SVMPrimal:
            current_axis.plot(x_linspace, (clss - clf.b - clf.w[0] * x_linspace) / clf.w[1], color='black', linestyle=':', label='margine')

    for i in range(len(support_vector_slacks)):
        current_axis.annotate("{:.2f}".format(support_vector_slacks[i]), (support_vectors[i, 0], support_vectors[i, 1]))

    current_axis.set_xlim(*x_lim)
    current_axis.set_ylim(*y_lim)

    # suppress duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def add_test_data_visualization(current_axis, x, y):
    for clss in np.unique(np.unique(y)):
        current_axis.scatter(*[x[y.squeeze() == clss, i] for i in range(2)], marker='x', s=100, linewidths=2,
                             label='test klasa {:d}'.format(int(clss)))


def choose_svm(svm_solver_type, X_train, y_train, kernel_foo, C=1.0, sigma=1.0):
    if svm_solver_type == SVMSolverType.Primal:
        clf = SVMPrimal(X_train, y_train, C)

    elif svm_solver_type == SVMSolverType.Dual:
        clf = SVMDual(X_train, y_train, C, kernel_foo=lambda x1, x2: kernel_foo(x1, x2, sigma=sigma))

    else:
        raise NotImplementedError("SVMSolverType: ", svm_solver_type, " is not supported.")

    return clf


def experiment(x, y, C=1.0, svm_solver_type=SVMSolverType.Dual, test_size=0.2, random_state=None, resolution=(100, 100), sigma=1.0):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    transforms = Pipeline([('scaler', StandardScaler())])  # normalization
    transforms.fit(X_train)

    if svm_solver_type == SVMSolverType.Primal:
        clf = SVMPrimal(transforms.transform(X_train), y_train, C)

    elif svm_solver_type == SVMSolverType.Dual:
        clf = SVMDual(transforms.transform(X_train), y_train, C, kernel_foo=lambda x1, x2: radial_basis_kernel(x1, x2, sigma=sigma))

    else:
        raise NotImplementedError("SVMSolverType: ", svm_solver_type, " is not supported.")

    print('Test accuracy = {:2.2f}'.format(accuracy_score(y_test, clf.predict(transforms.transform(X_test)))))

    dataset_area_class_visualization(transforms.transform(X_train), y_train,
                                     predict_foo=lambda background_points: clf.predict(background_points),
                                     resolution=resolution)
    # add_test_data_visualization(plt.gca(), transforms.transform(X_test), y_test)

    add_support_vector_visualization(plt.gca(), transforms.transform(X_train), y_train, clf)


def train_and_predict(x_train, y_train, x_test, svm_solver_type, C, sigma):
    transforms = Pipeline([('scaler', StandardScaler())])  # normalization
    transforms.fit(x_train)
    clf = choose_svm(svm_solver_type, transforms.transform(x_train), y_train, kernel_foo=radial_basis_kernel, C=C, sigma=sigma)

    return clf.predict(transforms.transform(x_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blobs', action=argparse.BooleanOptionalAction)
    parser.add_argument('--csv_data', action=argparse.BooleanOptionalAction)
    parser.add_argument('--svm_solver_type', type=str, required=True)

    args = parser.parse_args()
    print("Passed Arguments:\n", args)

    if args.svm_solver_type.lower() == 'primal':
        svm_solver_type = SVMSolverType.Primal
    elif args.svm_solver_type.lower() == 'dual':
        svm_solver_type = SVMSolverType.Dual
    else:
        raise ValueError("SVMSolverType '{:s}' is not supported.".format(args.svm_solver_type))

    if args.blobs:  # artificial dataset blobs
        for i in range(10):
            x, y = make_blobs(n_samples=100, centers=2, n_features=2)
            y = np.sign(y - 0.5)[:, np.newaxis]  # {-1, 1}

            experiment(x, y, C=1.0, svm_solver_type=svm_solver_type, test_size=0.2, random_state=78962, resolution=(100, 100), sigma=0.1)

    if args.csv_data:  # tabular data from memory
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        csv_path = os.path.join(__location__, 'svmData.csv')

        x, y = load_data(csv_path)

        experiment(x, y, C=1.0, svm_solver_type=svm_solver_type, test_size=0.2, random_state=78962, resolution=(100, 100), sigma=0.1)

        hyperparameter_search(x, y, start=-4, stop=2,
                              train_and_predict_foo=lambda x_train, y_train, x_test, C: train_and_predict(x_train, y_train, x_test,
                                                                                                          svm_solver_type, C, sigma=0.1),
                              metric_foo=accuracy_score, num=20, k_splits=5, n_repeats=2, confidence=0.95,
                              xlabel='C', ylabel='preciznost')

    plt.show()
