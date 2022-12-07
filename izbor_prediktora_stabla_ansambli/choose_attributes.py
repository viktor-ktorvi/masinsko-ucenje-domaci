import os

import numpy as np

from matplotlib import pyplot as plt

from data_loading.data_loading import load_data


def corrcoef(x1, x2):
    return np.corrcoef(x1.squeeze(), x2.squeeze())[0, 1]


if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path_1 = os.path.join(__location__, 'data_1.csv')

    np.random.seed(6122022)

    x, y = load_data(csv_path_1)
    num_features = x.shape[1]
    abs_corr_coefs = np.abs(np.array([corrcoef(x[:, i], y) for i in range(num_features)]))
    sorted_ids = np.argsort(abs_corr_coefs)

    plt.figure()
    x_axis = np.arange(num_features)
    plt.xticks(x_axis, sorted_ids)
    colors = np.random.rand(num_features, 3)
    for i in range(num_features):
        plt.plot(x_axis[i], abs_corr_coefs[sorted_ids[i]], marker='^', c=colors[i, :], markersize=16)  # marker=r'${:s}$'.format(str(sorted_ids[i]))

    plt.xlabel('redni broj prediktora')
    plt.ylabel(r'$|\rho(x_i, y)|$')
    plt.show()
