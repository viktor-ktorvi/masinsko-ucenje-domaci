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


if __name__ == '__main__':
    # x_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    # x_test = np.array([[10, 11], [12, 13], [14, 15]])
    x_train = np.random.randint(low=0, high=9, size=(6, 2))
    x_test = np.random.randint(low=0, high=9, size=(4, 2))

    norm_sqrd = cross_norm_sqrd(x_train, x_test)

    tau = 5
    weights = np.exp(-norm_sqrd / 2 / tau ** 2)

    # create 3d matrix of stacked diagonal matrices
    W = np.einsum('ij,jk->ijk', weights, np.eye(weights.shape[1], dtype=weights.dtype))
