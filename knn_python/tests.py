import unittest

import numpy as np

from knn import KNearestNeighbors
from tools import parse_data_file, split_x_data_and_class_vector
from os import path
import main
from matplotlib import pyplot as plt
from tools import class_vector_to_ints


class Tests(unittest.TestCase):

    def test_liver(self):
        data_x, class_vector = split_x_data_and_class_vector(parse_data_file("Data/bupa.data"), 6)
        class_vector, class_map = class_vector_to_ints(class_vector)
        error_rates_majority, error_rates_weighted = main.test_weighted_vs_majority(data_x, class_vector)
        plt.title("oi")
        main.plot_majority_vs_weighted(error_rates_majority, error_rates_weighted)
        plt.show()
        plt.clf()

    def test_knn(self):
        x_data = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [17, 18, 19]
        ])
        y = [1, 1, 2]
        knn = KNearestNeighbors(x_data, y, n_neighbors=2, weight_method='weighted', distance_method='euclidean')
        assert knn.classify(np.array([[18, 19, 20]]))[0] == 2

    def test_tools(self):
        data = parse_data_file(path.join("Data", "test.data"))
        expected = [
            ['1', '2', '3', '4', '5', 'test'],
            ['6', '7', '8', '9', '10', 'this works']
        ]
        assert data == expected
        data = parse_data_file(path.join("Data", "iris.data"))
        x, y = split_x_data_and_class_vector(data, 4)
        assert x.shape[0] == len(y)
        assert x.shape[1] + 1 == len(data[0])

    def test_bivariate_distribution(self):
        mean = [0, 0]
        cov = [[1, 0], [0, 100]]
        import matplotlib.pyplot as plt
        x, y = np.random.multivariate_normal(mean, cov, 5000).T
        plt.plot(x, y, 'x')
        plt.axis('equal')
        plt.show()
