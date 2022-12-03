import os

import numpy as np

from cvxopt import matrix, solvers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization


def train(x, y, C):
    """
    Solve a quadratic program to obtain the Support Vector Machine weights and bias.
    :param x: np.ndarray; shape num_samples x num_features; feature matrix
    :param y: np.ndarray; shape num_samples x 1; sample labels in {-1, 1}
    :param C: float; soft margin hyperparameter
    :return: tuple; weights, bias, boolean mask for support vectors
    """
    m, n = x.shape
    P = matrix((y * x) @ (y * x).T)
    q = matrix(-np.ones(m))
    A = matrix(y, (1, m))
    b = matrix(0.0)

    G = matrix(np.vstack((-np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))

    solution = solvers.qp(P, q, G, h, A, b)

    # alfa
    alfas = np.ravel(solution['x'])

    # support vectors
    sv_bool = alfas > 1e-5
    alfas_sv = alfas[sv_bool][:, np.newaxis]
    sv_x = x[sv_bool]
    sv_y = y[sv_bool]

    w = np.sum(alfas_sv * sv_y * sv_x, axis=0)
    b = np.mean(sv_y) - np.mean(sv_x, axis=0) @ w

    return w, b, sv_bool


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


def experiment(x, y, C=1.0, test_size=0.2, random_state=None, resolution=(100, 100)):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    transforms = Pipeline([('scaler', StandardScaler())])  # normalization
    transforms.fit(X_train)

    w, b, sv_bool = train(transforms.transform(X_train), y_train, C=C)
    dataset_area_class_visualization(transforms.transform(X_train), y_train,
                                     predict_foo=lambda background_points: predict(background_points, w, b),
                                     resolution=resolution)
    add_support_vector_visualization(plt.gca(), transforms.transform(X_train), y_train, sv_bool, w, b)


if __name__ == '__main__':

    experiment_type = {'blobs': True, 'csv_data': True}

    if experiment_type['blobs']:
        for i in range(10):
            x, y = make_blobs(n_samples=100, centers=2, n_features=2)
            y = np.sign(y - 0.5)[:, np.newaxis]

            experiment(x, y, C=1.0, test_size=0.2, random_state=78962, resolution=(100, 100))

    if experiment_type['csv_data']:
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        csv_path = os.path.join(__location__, 'svmData.csv')

        x, y = load_data(csv_path)

        experiment(x, y, C=1.0, test_size=0.2, random_state=78962, resolution=(100, 100))
    plt.show()
