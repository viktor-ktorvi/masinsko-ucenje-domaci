import os
import numpy as np
from matplotlib import pyplot as plt

from data_loading.data_loading import load_transformed_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.model import train_one_vs_rest

if __name__ == '__main__':
    epochs = 100
    lrs = [0.3, 0.3, 0.3, 1e-5, 1000]
    batch_sizes = [32, 1, 140, 32, 32]

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path = os.path.join(__location__, '../multiclass_data.csv')

    X_train_transformed, X_test_transformed, y_train, y_test = load_transformed_data(csv_path,
                                                                                     test_size=0.2, norm=True,
                                                                                     dimensionality=2)
    plt.figure()
    plt.title('Srednja verodostojnost po klasi')
    for i in range(len(lrs)):
        classifiers, loggers = train_one_vs_rest(X_train_transformed, y_train, epochs=epochs, lr=lrs[i],
                                                 batch_size=batch_sizes[i], log=True)

        mean_loss = np.mean(np.vstack([loggers[j].likelihood for j in range(len(loggers))]), axis=0)
        plt.plot(mean_loss, label="$\\alpha$ = {:g}, $m_{{mb}}$ = {:d}".format(lrs[i], batch_sizes[i]))

    plt.xlabel('epoha')
    plt.ylabel('verodostojnost')
    plt.legend()

    plt.show()
