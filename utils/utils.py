import numpy as np

from utils.validation import RepeatedKFold


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


def normalize(x, mean=None, std=None):
    """
    Normalize feature matrix x. If mean or std are None calculate them and return them along with the normalized feature
    matrix.
    :param x: np.ndarray; shape num_samples x num_features
    :param mean: np.ndarray; shape num_features x 1
    :param std: np.ndarray; shape num_features x 1
    :return: either np.ndarray of the same shape as x or a tuple of the same shape of (x, mean, std)
    """
    none_flag = False
    if mean is None:
        mean = np.mean(x, axis=0)
        none_flag = True

    if std is None:
        std = np.std(x, axis=0)
        none_flag = True

    assert np.count_nonzero(std) == len(std), 'Zero std encountered.'

    if none_flag:
        return (x - mean) / std, mean, std

    return (x - mean) / std


def sigmoid(x):
    """
    Sigmoid function.
    :param x: np.ndarray
    :return: np.ndarray; shape of x
    """
    return 1 / (1 + np.exp(-x))


class DataLoader:
    """
    Dataloader class. Returns shuffled indices arranged into batches.
    """

    def __init__(self, num_samples, batch_size, random_state=26848):
        """

        :param num_samples: int; size of dataset
        :param batch_size: int; batch size
        :param random_state: int; random seed for the random number generator
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        if self.num_samples % self.batch_size == 0:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = self.num_samples // self.batch_size + 1

        self.rng = np.random.default_rng(random_state)

    def get_batches(self):
        """
        Return shuffled indices split into batches.
        :return: list[np.ndarray]
        """
        indices = np.arange(self.num_samples)
        self.rng.shuffle(indices)

        batches = []
        for i in range(self.num_batches):
            if (i + 1) * self.batch_size < len(indices):
                batches.append(indices[i * self.batch_size:(i + 1) * self.batch_size])
            else:
                batches.append(indices[i * self.batch_size:])
                break

        assert len(batches) == self.num_batches

        running_sum = 0
        for b in batches:
            running_sum += len(b)

        assert running_sum == self.num_samples

        return batches


def running_average(x, window_size):
    """
    Average over a rectangular window.
    :param x: array like
    :param window_size: int
    :return: np.ndarray
    """
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')


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
