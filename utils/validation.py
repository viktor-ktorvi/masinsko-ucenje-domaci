import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
from tqdm import tqdm


def repeated_k_fold(x, y, train_and_predict_foo, metric_foo, n_splits=2, n_repeats=2, random_state=984613):
    """
    Do k-fold cross validation multiple times and return an error array for every model trained.

    :param x: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y: np.ndarray; shape num_samples x 1; dataset output vector
    :param train_and_predict_foo:
    :param metric_foo:
    :param n_splits: int; k in k-fold
    :param n_repeats: int; number of times to do k-fold cross validation
    :param random_state: int; random seed
    :return: metric matrix; np.ndarray; shape (n_splits * n_repeats) x 1
    """
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    metric = np.zeros((n_splits * n_repeats,), dtype=np.float32)

    for i, indices in enumerate(rkf.split(x)):
        train_index, test_index = indices
        x_train, x_test = x[train_index, :], x[test_index, :]
        y_train, y_test = y[train_index, :], y[test_index, :]

        y_pred = train_and_predict_foo(x_train, y_train, x_test)

        metric[i] = metric_foo(y_test, y_pred)

    return metric


class RepeatedKFold:
    """
    A class implementing part of sklearn.model_selection.RepeatedKFold's interface. Generates indices for k-fold cross
    validation repeated n times with different splits.
    """

    def __init__(self, n_splits=2, n_repeats=1, random_state=321653):
        """

        :param n_splits: int; k in k-fold
        :param n_repeats: int; number of times to do k-fold
        :param random_state: int; random seed
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats

        self.rng = np.random.default_rng(random_state)

    def split(self, x):
        """
        Generate folds for the training set x.
        :param x: np.ndarray; shape num_samples x num_features; dataset matrix
        :return: list[tuple(np.ndarray)]; a list of the train and test indices for every repeated split
        """
        validation_indices = []

        for i in range(self.n_repeats):
            indices = np.arange(x.shape[0])
            self.rng.shuffle(indices)
            folds = np.array_split(indices, self.n_splits)

            # get all n choose n - 1 combinations
            combinations = list(itertools.combinations(folds, self.n_splits - 1))

            for comb in combinations:
                train_idx = np.concatenate(comb)
                test_idx = np.setdiff1d(indices, train_idx, assume_unique=True)
                validation_indices.append((train_idx, test_idx))

        return validation_indices


def hyperparameter_search_metrics(x, y, hyperparameter_values, train_and_predict_foo, metric_foo, k_splits=5, n_repeats=1, confidence=0.95):
    """
    Hyperparameter search with repeated k-fold validation loop. Return the metrics and confidence interval.
    :param x: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y: np.ndarray; shape num_samples x 1; dataset output vector
    :param hyperparameter_values: List[float or int] or np.ndarray; the values of the hyperparameter.
    :param train_and_predict_foo: handle of function to train a classifier and predict on the test set
    :param metric_foo: handle of function that measures the classifier performance
    :param k_splits: int; k in k-folds cross validation
    :param n_repeats: int; number of times to repeat k-folds
    :param confidence: float; [0.0-1.0]; confidence interval to display
    :return: tuple(np.ndarray); metrics array of shape len_hyperparameter_values x (n_repeats * k_splits); confidence intervals array of shape len_hyperparameter_values x 2
    """
    metrics = np.zeros((len(hyperparameter_values), k_splits * n_repeats), dtype=np.float32)
    conf_interval = np.zeros((len(hyperparameter_values), 2), dtype=np.float32)

    for i in tqdm(range(len(hyperparameter_values))):
        metrics[i, :] = repeated_k_fold(x, y,
                                        train_and_predict_foo=lambda x_train, y_train, x_test: train_and_predict_foo(x_train, y_train,
                                                                                                                     x_test,
                                                                                                                     hyperparameter_values[
                                                                                                                         i]),
                                        metric_foo=metric_foo,
                                        n_splits=k_splits,
                                        n_repeats=n_repeats)

        conf_interval[i, :] = st.t.interval(confidence=confidence, df=len(metrics[i, :]) - 1,
                                            loc=np.mean(metrics[i, :]),
                                            scale=st.sem(metrics[i, :]))

    return metrics, conf_interval


def plot_hyperparameter_search(hyperparameter_values, conf_interval, metrics, confidence, k_splits, xlabel, ylabel):
    """
    Display the results of the hyperparameter search - the metric values plus the confidence interval.
    :param hyperparameter_values: List[float or int] or np.ndarray; the values of the hyperparameter
    :param conf_interval: np.ndarray; shape len_hyperparameter_values x 2; confidence intervals array
    :param metrics: np.ndarray; shape len_hyperparameter_values x (n_repeats * k_splits); metrics array
    :param k_splits: int; k in k-folds cross validation
    :param confidence: float; [0.0-1.0]; confidence interval to display
    :param xlabel: str; x label for plot
    :param ylabel: str; y label for plot
    :return: np.ndarray; metrics
    :return:
    """
    plt.figure()
    plt.fill_between(hyperparameter_values, conf_interval[:, 0], conf_interval[:, 1], color='paleturquoise', alpha=0.6,
                     label='interval poverenja {:d} %'.format(round(confidence * 100)))
    plt.plot(hyperparameter_values, np.mean(metrics, axis=1), '--^',
             label='srednja vrednost k ={:2d}\nvalidacionih podskupova'.format(k_splits))
    plt.grid(True, which="both", ls=":")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def log_hyperparameter_search(x, y, start, stop, train_and_predict_foo, metric_foo, num=1, k_splits=5, n_repeats=1, confidence=0.95,
                              xlabel='hyperparameter', ylabel='metric'):
    """
    Log(base 10) scale hyperparameter grid search.

    :param x: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y: np.ndarray; shape num_samples x 1; dataset output vector
    :param start: float; exponent of the start of the interval
    :param stop: float; exponent of the end of the interval
    :param train_and_predict_foo: handle of function to train a classifier and predict on the test set
    :param metric_foo: handle of function that measures the classifier performance
    :param num: int; number of points to search
    :param k_splits: int; k in k-folds cross validation
    :param n_repeats: int; number of times to repeat k-folds
    :param confidence: float; [0.0-1.0]; confidence interval to display
    :param xlabel: str; x label for plot
    :param ylabel: str; y label for plot
    :return: np.ndarray; metrics
    """

    hyperparameter_range = np.logspace(start=start, stop=stop, num=num)
    metrics, conf_interval = hyperparameter_search_metrics(x, y, hyperparameter_range, train_and_predict_foo, metric_foo, k_splits, n_repeats, confidence)

    plot_hyperparameter_search(hyperparameter_range, conf_interval, metrics, confidence, k_splits, xlabel, ylabel)
    plt.xscale('log')

    return metrics


def custom_hyperparameter_search(x, y, hyperparameter_values, train_and_predict_foo, metric_foo, k_splits=2, n_repeats=1, confidence=0.95,
                                 xlabel='hyperparameter', ylabel='metric'):
    """

    :param x: np.ndarray; shape num_samples x num_features; dataset feature matrix
    :param y: np.ndarray; shape num_samples x 1; dataset output vector
    :param hyperparameter_values: List[float or int] or np.ndarray; the values of the hyperparameter
    :param train_and_predict_foo: handle of function to train a classifier and predict on the test set
    :param metric_foo: handle of function that measures the classifier performance
    :param k_splits: int; k in k-folds cross validation
    :param n_repeats: int; number of times to repeat k-folds
    :param confidence: float; [0.0-1.0]; confidence interval to display
    :param xlabel: str; x label for plot
    :param ylabel: str; y label for plot
    :return: np.ndarray; metrics
    """
    metrics, conf_interval = hyperparameter_search_metrics(x, y, hyperparameter_values, train_and_predict_foo, metric_foo, k_splits, n_repeats, confidence)
    plot_hyperparameter_search(hyperparameter_values, conf_interval, metrics, confidence, k_splits, xlabel, ylabel)
    return metrics
