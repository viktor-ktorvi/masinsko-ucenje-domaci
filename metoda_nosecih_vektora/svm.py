import argparse
import os

import numpy as np

from cvxopt import matrix, solvers
from enum import IntEnum
from scipy.linalg import block_diag
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization


class SVMSolverType(IntEnum):
    Primal = 1
    Dual = 2


def train_primal(x, y, C):
    """
    Solve a quadratic program in primal from to obtain the Support Vector Machine weights and bias.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param y: np.ndarray; shape num_samples x 1; sample labels in {-1, 1}
    :param C: float; soft margin hyperparameter
    :return: tuple; weights, bias, boolean mask for support vectors
    """
    num_samples, num_features = x.shape
    P = block_diag(np.eye(num_features), np.zeros((num_samples + 1, num_samples + 1)))
    q = np.vstack((np.zeros((num_features + 1, 1)), C * np.ones((num_samples, 1))))

    G1 = np.hstack((-y * x, -y, -np.eye(num_samples)))
    h1 = -np.ones((num_samples, 1))

    G2 = np.hstack((np.zeros((num_samples, num_features + 1)), -np.eye(num_samples)))
    h2 = np.zeros((num_samples, 1))

    G = np.vstack((G1, G2))
    h = np.vstack((h1, h2))

    solution = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

    w = np.array(solution['x'][:num_features]).squeeze()
    b = np.array(solution['x'][num_features])

    alpha = np.array(solution['z'][:num_samples])
    sv_bool = alpha > 1e-5

    return w, b, sv_bool.squeeze()


def train_dual(x, y, C):
    """
    Solve a quadratic program in dual from to obtain the Support Vector Machine weights and bias.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param y: np.ndarray; shape num_samples x 1; sample labels in {-1, 1}
    :param C: float; soft margin hyperparameter
    :return: tuple; weights, bias, boolean mask for support vectors
    """
    num_samples, num_features = x.shape
    P = (y * x) @ (y * x).T
    q = -np.ones((num_samples, 1))
    A = y.reshape(1, num_samples)
    b = 0.0

    G = np.vstack((-np.eye(num_samples), np.eye(num_samples)))
    h = np.hstack((np.zeros(num_samples), np.ones(num_samples) * C))

    solution = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))

    # alfa
    alpha = np.array(solution['x'])

    # support vectors
    sv_bool = (alpha > 1e-5).squeeze()
    alpha_sv = alpha[sv_bool]
    sv_x = x[sv_bool]
    sv_y = y[sv_bool]

    weights = np.sum(alpha_sv * sv_y * sv_x, axis=0)
    bias = np.mean(sv_y) - np.mean(sv_x, axis=0) @ weights

    return weights, bias, sv_bool


def predict(x, w, b):
    """
    Predict the classes of samples using an SVM classifier.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param w: np.ndarray; shape num_features x 1; SVM weights
    :param b: np.ndarray; shape 1 x 1; SVM bias
    :return: np.ndarray; class predictions in {-1, 1}
    """
    return np.sign(x @ w + b)


def add_support_vector_visualization(current_axis, x, y, support_vector_ids, w, b):
    """
    Highlight support vectors, draw margins and annotate slack variable values.

    :param current_axis: axis object
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param y: np.ndarray; shape num_samples x 1; sample labels in {-1, 1}
    :param support_vector_ids: np.ndarray; shape num_samples x 1; boolean mask for support vectors
    :param w: np.ndarray; shape num_features x 1; SVM weights
    :param b: np.ndarray; shape 1 x 1; SVM bias
    :return:
    """
    support_vectors = x[support_vector_ids]
    support_vector_labels = y[support_vector_ids].squeeze()

    support_vector_projection = support_vectors @ w + b
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

        current_axis.plot(x_linspace, (clss - b - w[0] * x_linspace) / w[1], color='black', linestyle=':', label='margine')

    for i in range(len(support_vector_slacks)):
        current_axis.annotate("{:.3f}".format(support_vector_slacks[i]), (support_vectors[i, 0], support_vectors[i, 1]))

    current_axis.set_xlim(*x_lim)
    current_axis.set_ylim(*y_lim)

    # suppress duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def experiment(x, y, C=1.0, svm_solver_type=SVMSolverType.Dual, test_size=0.2, random_state=None, resolution=(100, 100)):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    transforms = Pipeline([('scaler', StandardScaler())])  # normalization
    transforms.fit(X_train)

    if svm_solver_type == SVMSolverType.Primal:
        w, b, sv_bool = train_primal(transforms.transform(X_train), y_train, C=C)
    elif svm_solver_type == SVMSolverType.Dual:
        w, b, sv_bool = train_dual(transforms.transform(X_train), y_train, C=C)
    else:
        raise NotImplementedError("SVMSolverType: ", svm_solver_type, " is not supported.")

    dataset_area_class_visualization(transforms.transform(X_train), y_train,
                                     predict_foo=lambda background_points: predict(background_points, w, b),
                                     resolution=resolution)
    add_support_vector_visualization(plt.gca(), transforms.transform(X_train), y_train, sv_bool, w, b)


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

            experiment(x, y, C=1.0, svm_solver_type=svm_solver_type, test_size=0.2, random_state=78962, resolution=(100, 100))

    if args.csv_data:  # tabular data from memory
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        csv_path = os.path.join(__location__, 'svmData.csv')

        x, y = load_data(csv_path)

        experiment(x, y, C=1.0, svm_solver_type=svm_solver_type, test_size=0.2, random_state=78962, resolution=(100, 100))
    plt.show()
