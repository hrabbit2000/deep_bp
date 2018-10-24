# -*- coding: UTF-8 -*-

import numpy as np
import mnist_loader as loader
import deep_bp as dbp

def main():
    training_data, validation_data, test_data = loader.load_data()
    bp = dbp.DeepBP([784, 30, 10])
    bp.init_parameters()
    # bp.weights = [np.array([(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]), np.array([(0.5, 0.6, 0.7)])]
    # bp.biases = [np.array([(0.3, 0.4, 0.5)]), np.array([(0.2,)])]
    # inputs = [np.array([(1, 2), (2, 3)])]
    # ds = [np.array([(0.5), (0.6)])]
    print(bp.weights[1][0])
    print("*********************************************************************")
    bp.sgd(training_data, 30, 10, 2)
    print(bp.weights[1][0])


if __name__ == "__main__": main()
