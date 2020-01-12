from tools import split_x_data_and_class_vector, parse_data_file, class_vector_to_ints
import os
from os import path
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from knn import KNearestNeighbors
import numpy as np
from typing import Dict, Tuple
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sys import argv

DATA_DIR = 'Data'
IRIS_DATA = path.join(DATA_DIR, 'iris.data')
DERMATOLOGY_DATA = path.join(DATA_DIR, 'dermatology.data')
LANDSAT_DATA = path.join(DATA_DIR, "sat.trn")
BANKNOTE_DATA = path.join(DATA_DIR, 'data_banknote_authentication.data')
RESULT_DIR = 'Results'


def test_weighted_vs_majority(x_data, class_vector) -> (Dict, Dict):
    """
    :return: Returns (error_rates_majority, error_rates_weighted)
    """
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    average_error_rates: Dict[Tuple[str, int], float] = {}

    for weight_method in KNearestNeighbors.WEIGHT_METHODS:
        for n_neighbors in range(1, 51):
            classification_errors = []
            for train_indexes, test_indexes in kfold.split(x_data, class_vector):
                knn = KNearestNeighbors(x_data[train_indexes], class_vector[train_indexes], weight_method=weight_method,
                                        n_neighbors=n_neighbors)
                predicteds = knn.classify(x_data[test_indexes])
                n_errors = 0

                for predicted, actual in zip(predicteds, class_vector[test_indexes]):
                    if predicted != actual:
                        n_errors += 1

                classification_errors.append(n_errors / len(predicteds))

            average_error_rates[(weight_method, n_neighbors)] = np.average(classification_errors)

    def error_rates_method(method):
        return {n_n: err_rate for (method_, n_n), err_rate in average_error_rates.items() if method_ == method}

    return error_rates_method('majority'), error_rates_method('weighted')


def plot_majority_vs_weighted(error_rates_majority, error_rates_weighted):
    plt.ylabel("average error")
    plt.xlabel("number of neighbors")
    plt.plot(list(error_rates_weighted.keys()), list(error_rates_weighted.values()), label="weighted KNN")
    plt.plot(list(error_rates_majority.keys()), list(error_rates_majority.values()), label="majority KNN")
    plt.legend()


def main():
    global RESULT_DIR, IRIS_DATA, DATA_DIR, DERMATOLOGY_DATA, LANDSAT_DATA

    while not path.isdir(RESULT_DIR):
        if not path.exists(RESULT_DIR):
            os.mkdir(RESULT_DIR)
        else:
            RESULT_DIR += '_'

    lock = Lock()

    def data_file_test(data_file_path, class_column, title):
        data_x, class_vector = split_x_data_and_class_vector(parse_data_file(data_file_path), class_column)
        class_vector, class_map = class_vector_to_ints(class_vector)
        error_rates_majority, error_rates_weighted = test_weighted_vs_majority(data_x, class_vector)
        lock.acquire()
        plt.title(title)
        plot_majority_vs_weighted(error_rates_majority, error_rates_weighted)
        title = title.replace('\n', ' ')
        plt.savefig(path.join(RESULT_DIR, f"{title}.png"))
        plt.clf()
        lock.release()

    def iris_data_test():
        """
        Test with the iris data set from: https://archive.ics.uci.edu/ml/datasets/iris
        """
        data_file_test(IRIS_DATA, 4, "Comparison of majority KNN and weighted KNN for the iris data set")

    def dudani_paper_test(n_classes):
        """
        Test with the bivariate class distribution of the Dudani paper using 3000 samples:
        Dudani, Sahibsingh A. "The distance-weighted k-nearest-neighbor rule." IEEE Transactions on Systems, Man, and Cybernetics 4 (1976): 325-327.
        """
        classes_1 = np.concatenate(
            (np.random.multivariate_normal([3, 3], [[2.25, 0], [0, 2.25]], int(n_classes * (3 / 5))),
             np.random.multivariate_normal([7, 7], [[2.25, 0], [0, 2.25]],
                                           int(n_classes * (2 / 5))))
        )
        classes_2 = np.random.multivariate_normal([4, 6], [[4, 0], [0, 4]], n_classes)
        classes_3 = np.random.multivariate_normal([7.3, 3.5], [[9, 0], [0, 9]], n_classes)
        class_vector = np.array([1] * classes_1.shape[0] + [2] * classes_2.shape[0] + [3] * classes_3.shape[0])
        x_data = np.concatenate((classes_1, classes_2, classes_3))

        error_rates_majority, error_rates_weighted = test_weighted_vs_majority(x_data, class_vector)
        lock.acquire()
        title = f"Class distribution of the data set in the Dudani paper\nwith {n_classes} classes per class"
        plt.title(title)
        plt.scatter(classes_1[:, 0], classes_1[:, 1], label='Class 1')
        plt.scatter(classes_2[:, 0], classes_2[:, 1], label='Class 2')
        plt.scatter(classes_3[:, 0], classes_3[:, 1], label='Class 3')
        plt.xlabel("Attribute 1")
        plt.ylabel("Attribute 2")
        plt.legend()
        title = title.replace('\n', ' ')
        plt.savefig(path.join(RESULT_DIR, f"{title}.png"))
        plt.clf()

        title = f"Comparison of majority KNN and weighted KNN\nfor data set in the Dudani paper with {n_classes} classes per class"
        plt.title(title)
        plot_majority_vs_weighted(error_rates_majority, error_rates_weighted)
        title = title.replace('\n', ' ')
        plt.savefig(path.join(RESULT_DIR, f"{title}.png"))

        plt.clf()
        lock.release()



    def dermatology_test():
        """
        Test with dermatology data set, using only the entries which are complete.
        From: https://archive.ics.uci.edu/ml/datasets/dermatology
        """
        data_file_test(DERMATOLOGY_DATA, 34,
                       "Comparison of majority KNN and weighted KNN\nfor the dermatology data set")

    def landsat_test():
        """
        Test with landsat data set from https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/
        """
        data_file_test(LANDSAT_DATA, 36, "Comparison of majority KNN and weighted KNN\nfor the landsat data set")

    def banknote_authentication_test():
        """
        Test with banknote authentication data set from http://archive.ics.uci.edu/ml/datasets/banknote+authentication
        """
        data_file_test(BANKNOTE_DATA, 4, "Comparison of majority KNN and weighted KNN\nfor the banknote data set")

    # Different test cases
    tasks = (
        lambda: dudani_paper_test(1000),
        lambda: dudani_paper_test(100),
        landsat_test,
        banknote_authentication_test,
        iris_data_test,
        dermatology_test,
    )

    if "--multithread" in argv:
        executor = ThreadPoolExecutor(max_workers=min(cpu_count(), len(tasks)))
        results = []
        for task in tasks:
            executor.submit(task)
        for result in results:
            result.get()
        executor.shutdown(wait=True)
    else:
        for task in tasks:
            task()


if __name__ == '__main__':
    main()
