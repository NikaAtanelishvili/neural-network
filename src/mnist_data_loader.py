import gzip
import pickle

import numpy as np

def load_mnist_data(file_path, chunk_size=5000):
    with gzip.open(file_path, 'rb') as f:
        tr_d, _, te_d  = pickle.load(f, encoding='latin1')  # Use 'latin1' for compatibility

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = np.array(list(zip(training_inputs, training_results)), dtype=object)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs[:chunk_size], te_d[1]))

    return training_data, test_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e