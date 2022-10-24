import itertools
import numpy as np


def cross_norm_sqrd(x_train, x_test):
    """
    Calculate the L2 norm between every vector in the two matrices.
    :param x_train: np.ndarray; shape N x n
    :param x_test: np.ndarray; shape M x n
    :return: np.ndarray of L2 squared norms between every vector in the 2 matrices; shape M x N
    """

    # for every vector in train create a copy of test
    x_test_3d = np.repeat(x_test[:, np.newaxis, :], x_train.shape[0], axis=1)

    difference_sqrd = (x_train - x_test_3d) ** 2
    return np.sum(difference_sqrd, axis=2)


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


if __name__ == '__main__':
    # cross_norm_sqrd

    x_train = np.random.randint(low=0, high=9, size=(123, 2))
    x_test = np.random.randint(low=0, high=9, size=(17, 2))

    norm_sqrd = cross_norm_sqrd(x_train, x_test)

    tau = 5
    weights = np.exp(-norm_sqrd / 2 / tau ** 2)

    # create 3d matrix of stacked diagonal matrices
    W = np.einsum('ij,jk->ijk', weights, np.eye(weights.shape[1], dtype=weights.dtype))

    # RepeatedKFold
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=321653)
    rkf.split(x_train)
