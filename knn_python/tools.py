import numpy as np
from typing import Any, List, Dict
import re


def parse_data_file(file_path: str) -> List[List[Any]]:
    """
    Reads the file at file_path
    :param file_path:
    :return:
    """
    if file_path.endswith(".data"):
        split = ","
    elif file_path.endswith(".trn"):
        split = " "
    else:
        raise Exception("unrecognized file")
    with open(file_path) as f:
        # split the lines accordingly
        data = [line.rstrip().split(split) for line in f.readlines() if re.match("\\S+", line) and '?' not in line]
    assert is_2darray_compatible(data), "amount of attributes is inconsistent in file"
    return data


def is_2darray_compatible(list2d: List[List[Any]]) -> bool:
    """
    :param list2d:
    :return: Checks whether a list is NxM
    """
    for index, item in enumerate(list2d):
        if index - 1 >= 0 and len(list2d[index - 1]) != len(item):
            return False
    return True


def split_x_data_and_class_vector(data: List[List[Any]], class_col: int) -> (np.ndarray, np.ndarray):
    """
    Split the data into a class vector and attributes
    :param data:
    :param class_col: the columns in data which indicates the class
    :return: if data is NxM, then returns Nx(M-1) array containing the attributes, and an Nx1 array containing the classes
    """
    assert is_2darray_compatible(data), "each sample in the data should have the same amount of attributes"
    if data:
        attr_amount = len(data[0])
        x_result = np.zeros((len(data), attr_amount - 1))
        assert class_col < attr_amount, "Class column should be within the data"
        y_result = []
        for row, sample in enumerate(data):
            for col, attr in enumerate(sample):
                if col == class_col:
                    y_result.append(attr)
                else:
                    x_result[row, col] = float(attr)
        return x_result, np.array(y_result)
    return np.zeros((0, 0)), np.zeros(0)


def class_vector_to_ints(class_vector: np.ndarray) -> (np.ndarray, Dict[int, Any]):
    """
    Maps a class vector of any hashable type to a class vector of integers. The integers represent the class an index belongs to.
    :param class_vector:
    :return: class_vector of ints, dictionary which holds the classes corresponding to the ints
    """
    int_cls_vect = []
    new_class = 1
    class_map = {}
    for item in class_vector:
        if item in class_map.keys():
            int_cls_vect.append(class_map[item])
        else:
            class_map[item] = new_class
            int_cls_vect.append(new_class)
            new_class += 1
    return np.array(int_cls_vect, dtype=int), {int_: class_ for class_, int_ in class_map.items()}
