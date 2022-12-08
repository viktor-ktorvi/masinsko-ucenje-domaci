import os

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from matplotlib import pyplot as plt

from utils.validation import custom_hyperparameter_search

from data_loading.data_loading import load_data
from generalizovani_linearni_modeli_i_generativni_algoritmi.logisticka_regresija.visualization import \
    dataset_area_class_visualization

if __name__ == '__main__':
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    csv_path_1 = os.path.join(__location__, 'data_1.csv')  # data path

    random_state = 8122022
    # np.random.seed(random_state)  # for reproducibility

    x, y = load_data(csv_path_1)  # load data

    selected_feature_ids = [11, 2]  # select 2 features
    x_selected = x[:, selected_feature_ids]

    criterion = "entropy"
    splitter = "best"


    def train_and_predict_foo(x_train, y_train, x_test, max_depth):
        """
        Train a decision tree and predict on the test set.
        """
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, criterion=criterion, splitter=splitter)
        clf.fit(x_train, y_train)

        return clf.predict(x_test)


    # search the space of max depth
    custom_hyperparameter_search(x_selected, y, np.arange(1, 40), train_and_predict_foo=train_and_predict_foo, metric_foo=accuracy_score,
                                 k_splits=5, n_repeats=2, confidence=0.95, xlabel='maksimalna dubina stabla', ylabel='tačnost')

    x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=random_state)

    for depth in [1, 4, 20]:  # plot under-fitting, just right, over-fitting
        clf = DecisionTreeClassifier(max_depth=depth, random_state=random_state, criterion=criterion, splitter=splitter)
        clf.fit(x_train, y_train)

        dataset_area_class_visualization(x_selected, y,
                                         predict_foo=lambda background_points: clf.predict(background_points),
                                         resolution=(200, 200))
    plt.show()
