import os
import pickle
import numpy as np


def get_available_path(_dir, _file):
    os.makedirs(_dir, exist_ok=True)
    if not os.path.exists(_file):
        _file = os.path.join(_dir, _file)
    return _file


def pickle_load(path):
    if not os.path.exists(path):
        return []
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_dump(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def stable_softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator