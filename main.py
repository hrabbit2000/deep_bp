# -*- coding: UTF-8 -*-

import numpy as np
import mnist_loader as loader
import deep_bp as dbp

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10))
    e[j] = 1.0
    return e

def wrap_data():
    tr_d, va_d, te_d = loader.load_data()
    training_inputs = [x for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    return training_data, None, None

def main():
    training_data, validation_data, test_data = wrap_data()
    bp = dbp.DeepBP([784, 30, 10], False)
    bp.init_parameters()
    bp.weights = [np.array([(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]), np.array([(0.5, 0.6, 0.7)])]
    bp.biases = [np.array([(0.3, 0.4, 0.5)]), np.array([(0.2,)])]
    inputs = [np.array([(1, 2), (2, 3)])]
    ds = [np.array([(0.5), (0.6)])]
    # bp.sgd(training_data, 30, 10, 2)
    bp.batch_process(inputs, ds, 0.5)
    print(bp.weights)
    print(bp.biases)


if __name__ == "__main__": main()
