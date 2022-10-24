import numpy as np
import scipy.stats as st

from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.metrics import mean_squared_error

from data_loading import load_data, add_bias
from utils import cross_norm_sqrd, RepeatedKFold


def get_weights(x_test, x_train, tau):
    """
    For every test sample get a diagonal matrix of weights with respect to the training set.
    The number of features is denoted my n
    :param x_test:  np.ndarray; shape M x n; test samples
    :param x_train: np.ndarray; shape N x n; train samples
    :param tau: float; standard deviation of the gaussian kernel
    :return: np.ndarray; shape M x N x N; 3d matrix of stacked diagonal matrices of weights
    """
    norm_sqrd = cross_norm_sqrd(x_train, x_test)  # M x N

    weights = np.exp(-norm_sqrd / 2 / tau ** 2)  # M x N

    # create 3d matrix of M stacked diagonal matrices(N x N)
    return np.einsum('ij,jk->ijk', weights, np.eye(weights.shape[1], dtype=weights.dtype))  # M x N x N


def regress(x_test, x_train, weights, y_train):
    """
    Calculate linear weights theta of the weighted linear regression for the entire test dataset without using
    explicit python for loops. Applies those weights on the test dataset and returns predictions.
    :param x_test:
    :param x_train:
    :param weights:
    :param y_train:
    :return:
    """
    xtw = np.einsum('ij, kjl->kil', x_train.T, weights)  # M x n x N
    xtwx = np.einsum('ijk, kl->ijl', xtw, x_train)  # M x n x n
    inv = np.linalg.inv(xtwx)  # M x n x n

    xtwy = np.einsum('ijk, k->ij', xtw, y_train.squeeze())  # M x n

    theta = np.einsum('ijk, ik->ij', inv, xtwy)  # M x n

    return np.einsum('ij, ij->i', x_test, theta)  # M x 1


def lwlr(x_test, x_train, y_train, tau=0.2):
    """
    Add the bias term to feature vectors and perform locally weighted linear regression.

    :param x_test: np.ndarray; shape M x num_features
    :param x_train: np.ndarray; shape N x num_features
    :param y_train: np.ndarray; shape N x 1
    :param tau: float; hyperparameter
    :return: np.ndarray; shape M x 1; test predictions
    """
    x_test_biased = add_bias(x_test)
    x_train_biased = add_bias(x_train)
    # TODO mozda normalizovati jer se koriste distance

    W = get_weights(x_test=x_test_biased, x_train=x_train_biased, tau=tau)
    return regress(x_test=x_test_biased, x_train=x_train_biased, weights=W, y_train=y_train)


def example_1d():
    """
    Locally weighted linear regression 1D example.
    :return:
    """
    start = -2 * np.pi
    stop = 2 * np.pi
    num_1d = 500
    x_1d = np.linspace(start=start, stop=stop, num=num_1d)
    x_1d_test = np.random.uniform(low=start, high=stop, size=(num_1d // 4, 1))
    y_1d = np.sin(x_1d) + np.sin(2 * x_1d) + 0.3 * np.random.randn(len(x_1d))

    x_1d = x_1d.reshape(x_1d.shape[0], 1)
    y_1d = y_1d.reshape(y_1d.shape[0], 1)

    y_1d_test = lwlr(x_1d_test, x_1d, y_1d, tau=0.2)

    fig1d = plt.figure()
    plt.grid()
    plt.plot(x_1d, y_1d, 'o', label='trening')
    plt.plot(x_1d_test, y_1d_test, 'rx', label='test')
    plt.title('Primer 1D')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()


def example_2d():
    """
    Locally weighted linear regression 2D example.
    :return:
    """
    start = -2 * np.pi
    stop = 2 * np.pi
    nx, ny = (40, 40)

    X1, X2 = np.meshgrid(np.linspace(start, stop, nx), np.linspace(start, stop, ny))
    x_2d = np.hstack((X1.flatten().reshape(nx * ny, 1), X2.flatten().reshape(nx * ny, 1)))

    z = np.cos(X1) + np.cos(X2) + 0.1 * np.random.randn(nx, ny)
    y_2d = z.flatten()

    nx_test = nx // 4
    ny_test = ny // 4
    x_2d_test = np.random.uniform(low=start, high=stop, size=(nx_test * ny_test, 2))
    # X1_test, X2_test = np.meshgrid(x_2d_test[:, 0], x_2d_test[:, 1])

    y_2d_test = lwlr(x_2d_test, x_2d, y_2d, tau=0.2)

    fig2d = plt.figure()
    ax = fig2d.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X1, X2, z, alpha=0.3, label='trening')

    # code to fix labeling error https://stackoverflow.com/questions/54994600/pyplot-legend-poly3dcollection-object-has-no-attribute-edgecolors2d
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    ax.scatter(x_2d_test[:, 0], x_2d_test[:, 1], y_2d_test, color='r', label='test')

    ax.set_title('Primer 2D')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    ax.legend()


def model(x):
    """
    A function that predicts the values y with respect to the given vectors x using the predetermined dataset and
    locally weighted linear regression.
    :param x: np.ndarray; shape num_samples x num_features
    :return: np.ndarray; shape num_samples x 1
    """
    x_train, y_train = load_data()

    return lwlr(x, x_train, y_train, tau=0.2)


def repeated_k_fold(x, y, tau, n_splits=2, n_repeats=2, random_state=984613):
    """
    Do k-fold cross validation multiple times and return an error array for every model trained.

    :param x: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y: np.ndarray; shape num_samples x 1; dataset output vector
    :param tau: float; locally weighted linear regression hyperparameter
    :param n_splits: int; k in k-fold
    :param n_repeats: int; number of times to do k-fold cross validation
    :param random_state: int; random seed
    :return: error matrix; np.ndarray; shape (n_splits * n_repeats) x 1
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    rms_errors = np.zeros((n_splits * n_repeats,), dtype=np.float32)

    for i, indices in enumerate(rkf.split(x)):
        train_index, test_index = indices
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_pred = lwlr(x_test, x_train, y_train, tau)

        rms_errors[i] = np.sqrt(mean_squared_error(y_test, y_pred))

    return rms_errors


def hyperparameter_log_search(x, y, start, stop, num=1, k_splits=2, n_repeats=1, confidence=0.95):
    """
    Log(base 10) scale hyperparameter grid search for tau in the locally weighted linear regression.

    :param x: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y: np.ndarray; shape num_samples x 1; dataset output vector
    :param start: float; exponent of the start of the interval
    :param stop: float; exponent of the end of the interval
    :param num: int; number of points to search
    :param k_splits: int; k in k-folds cross validation
    :param n_repeats: int; number of times to repeat k-folds
    :param confidence: float; [0.0-1.0]; confidence interval to display
    :return:
    """
    rms_errors = np.zeros((num, k_splits * n_repeats), dtype=np.float32)
    conf_interval = np.zeros((num, 2), dtype=np.float32)
    taus = np.logspace(start=start, stop=stop, num=num)
    for i in tqdm(range(len(taus))):
        rms_errors[i, :] = repeated_k_fold(x, y, taus[i], n_splits=k_splits, n_repeats=n_repeats)
        conf_interval[i, :] = st.t.interval(confidence=confidence, df=len(rms_errors[i, :]) - 1,
                                            loc=np.mean(rms_errors[i, :]),
                                            scale=st.sem(rms_errors[i, :]))

    fig = plt.figure()
    plt.fill_between(taus, conf_interval[:, 0], conf_interval[:, 1], color='paleturquoise', alpha=0.6,
                     label='interval poverenja {:d} %'.format(round(confidence * 100)))
    plt.plot(taus, np.mean(rms_errors, axis=1), '--^',
             label='srednja vrednost k ={:2d}\nvalidacionih podskupova'.format(k_splits))
    plt.grid(True, which="both", ls=":")
    plt.xscale('log')
    plt.xlabel(r'$\tau$')
    plt.ylabel('rmse')
    plt.title(r'Koren srednje kvadratne greske u funkciji hiperparametra $\tau$')
    plt.legend()


if __name__ == '__main__':
    show_example = {'1d': False, '2d': False}
    if show_example['1d']:
        example_1d()

    if show_example['2d']:
        example_2d()

    x, y = load_data()

    hyperparameter_log_search(x, y, start=-2, stop=0, num=10, k_splits=5, n_repeats=2, confidence=0.95)

    plt.show()
