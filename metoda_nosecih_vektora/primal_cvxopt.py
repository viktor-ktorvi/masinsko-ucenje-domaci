import os

from cvxopt import matrix, solvers
import numpy as np

from scipy.linalg import block_diag
from skimage import filters
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.report_visual import dataset_visualization
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization


def predict(X, w, b):
    """
    Return class predictions.
    :param X: np.ndarray; shape num_samples x num_features; feature matrix.
    :param w: np.ndarray; shape num_features x 1; classifier weights
    :param w: np.ndarray; shape 1 x 1; classifier bias
    :return: np.ndarray; shape num_samples x 1; predictions in {-1, 1}
    """
    return np.sign(X @ w + b)


def train_svm_v2(x_train, y_train, C, verbose=False):
    num_samples = x_train.shape[0]

    P = matrix((y_train * x_train) @ (y_train * x_train).T)

    q = matrix(-np.ones(num_samples))
    A = matrix(y_train, (1, num_samples))
    b = matrix(0.0)

    G = matrix(np.vstack((np.diag(-np.ones(num_samples)), np.eye(num_samples))))
    h = matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * C)))

    sol = solvers.qp(P, q, G, h, A, b)
    # P22 = np.zeros((num_samples, num_samples))
    #
    # P12 = np.diag(y_train.squeeze())
    #
    # P = np.vstack((np.hstack((P11, P12)), np.hstack((P12, P22))))
    #
    # q = np.vstack((np.zeros((num_samples, 1)), C * np.ones((num_samples, 1))))
    #
    # G = np.hstack((np.zeros((num_samples, num_samples)), -np.eye(num_samples)))
    # h = np.zeros((num_samples, 1))
    #
    # print(np.linalg.matrix_rank(np.vstack((P, G))))
    #
    # solvers.options['show_progress'] = False
    # sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

    a = np.array(sol['x'])
    slacks = np.zeros(num_samples)

    w = np.sum(a * y_train * x_train, axis=0)

    a_filter = ((0 <= a) * (a <= C)).squeeze()
    b = y_train[a_filter][-1] - np.sum(a[a_filter] * y_train[a_filter] * x_train[a_filter] @ x_train[a_filter][-1].T)

    return w, b, a, slacks


def train_svm(x_train, y_train, C, verbose=False):
    num_samples = x_train.shape[0]
    num_features = x_train.shape[1]

    P = block_diag(np.eye(num_features), np.zeros((num_samples + 1, num_samples + 1)))  # w.T @ w
    q = np.vstack((np.zeros((num_features + 1, 1)), C * np.ones((num_samples, 1))))  # C * sum(slacks)

    G1 = np.hstack((-y_train * x_train, -y_train, -np.ones((num_samples, num_samples))))  # y_t * (X @ w + b) - 1 + slacks >= 0
    h1 = -np.ones((num_samples, 1))

    G2 = np.hstack((np.zeros((num_samples, num_features + 1)), -np.eye(num_samples)))  # slacks >= 0
    h2 = np.zeros((num_samples, 1))

    G = np.vstack((G1, G2))
    h = np.vstack((h1, h2))

    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

    # extract solutions
    sol_x = np.array(sol['x'])
    w = sol_x[:num_features]
    b = sol_x[num_features]
    slacks = sol_x[num_features + 1:]
    smv_inequality_lagrange_mul = np.array(sol['z'])[:num_samples]

    if verbose:
        print("Optimization status: {:s}".format(sol['status']))

    return w, b, smv_inequality_lagrange_mul, slacks


def add_support_vector_visualization(current_axis, x_transform, y, support_vector_ids, slack_vars):
    support_vectors = x_transform[support_vector_ids]
    support_vector_labels = y[support_vector_ids].squeeze()
    support_vector_slacks = slack_vars[support_vector_ids].squeeze()

    for clss in np.unique(support_vector_labels):
        current_axis.scatter(*[support_vectors[support_vector_labels == clss, i] for i in range(2)],
                             facecolor='none',
                             edgecolor='lime',
                             linewidths=2,
                             label='noseći vektori')

    for i in range(len(support_vector_slacks)):
        current_axis.annotate(r"$\xi = ${:.3f}".format(support_vector_slacks[i]), (support_vectors[i, 0], support_vectors[i, 1]))

    # suppress duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


if __name__ == '__main__':
    # __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # csv_path = os.path.join(__location__, 'svmData.csv')
    #
    # x, y = load_data(csv_path)

    test_size = 0.2
    random_state = 357862
    C = 1  # soft margin hyperparameter

    for i in range(10):
        x, y = make_blobs(n_samples=100, centers=2, n_features=2)
        y = np.sign(y - 0.5)[:, np.newaxis]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        pipe = Pipeline([('scaler', StandardScaler())])  # normalization
        pipe.fit(X_train)

        weights, bias, svm_inequality_lm, slack_vars = train_svm_v2(pipe.transform(X_train), y_train, C, verbose=True)

        threshold = filters.threshold_otsu(svm_inequality_lm)
        support_vector_ids = (svm_inequality_lm > threshold).squeeze()

        # y_pred = predict(pipe.transform(X_test), weights, bias)
        # print("Test accuracy = {:2.2f}".format(accuracy_score(y_test, y_pred)))

        dataset_area_class_visualization(pipe.transform(X_train), y_train,
                                         predict_foo=lambda background_points: predict(background_points, weights, bias),
                                         resolution=(100, 100))

        # add_support_vector_visualization(plt.gca(), pipe.transform(X_train), y_train, support_vector_ids, slack_vars)

        current_axis = plt.gca()

    plt.show()

    kjkszpj = None
