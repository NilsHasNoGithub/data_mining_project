from knn_rust import KNearestNeighbors as KNNRust
from typing import Iterable, List

import numpy as np


class KNearestNeighbors:
    """
    Wrapper python class for the rust implementation
    """
    WEIGHT_METHODS = ("weighted", "majority")
    DISTANCE_METHODS = ("manhattan", "euclidean")

    def __init__(self, x_data: np.ndarray, class_vector: List[int], n_neighbors=5, weight_method="majority",
                 distance_method="euclidean"):
        data_list = []
        for row in range(x_data.shape[0]):
            data_list.append([])
            for col in range(x_data.shape[1]):
                data_list[-1].append(float(x_data[row, col]))

        if len(data_list) != len(class_vector):
            raise Exception(
                f"The length of the class vector should be the same as the length of the x_data, but it wasnt."
                f" Length of class_vector: {len(class_vector)}, length of x_data: {len(x_data)}")

        self.attribute_amount = x_data.shape[1]

        self.knn_rust = KNNRust(x_data, class_vector)
        if n_neighbors >= 1:
            self.knn_rust.set_n_neighbors(n_neighbors)
        else:
            raise Exception(
                f"Number of neighbros used for classification should be greater or equal to one but was: {n_neighbors}")

        if weight_method.lower() in KNearestNeighbors.WEIGHT_METHODS:
            self.knn_rust.set_weight_method(weight_method)
        else:
            raise Exception(
                f"Weight method should either be in {KNearestNeighbors.WEIGHT_METHODS}, but was: {repr(weight_method)}")

        if distance_method.lower() in KNearestNeighbors.DISTANCE_METHODS:
            self.knn_rust.set_distance_method(distance_method)
        else:
            raise Exception(
                f"Distance method should be in {KNearestNeighbors.DISTANCE_METHODS} but was: {repr(distance_method)}")

    def classify(self, queries: np.ndarray) -> List[int]:

        results = []
        for row in range(queries.shape[0]):
            results.append(self.knn_rust.classify(list(queries[row, :])))
        return results
