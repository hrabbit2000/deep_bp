# -*- coding: UTF-8 -*-

import numpy as np
import time
import mnist_loader as loader
import deep_bp as dbp

print("trainning, please waiting...")

def main():
    training_data, validation_data, test_data = loader.load_data_wrapper()
    bp = dbp.DeepBP([784, 30, 10])
    t = time.time()
    bp.sgd(training_data, 1, 10, 3.0, test_data)
    print("time comsume: {0} / {1}".format(bp.delta_t2, time.time() - t))
    
    # bp = dbp.DeepBP([2, 3, 2], False)
    # bp.weights = [np.array([(0.1, 0.2, 0.3), (0.2, 0.3, 0.4)]), np.array([(0.5, 1), (0.6, 1), (0.7, 1)])]
    # bp.biases = [np.array([(0.3,), (0.4,), (0.5,)]), np.array([(0.2,), (1,)])]
    # inputs = [np.array([(1,), (2,)])]
    # ds = [np.array([(0.5,), (0.2,)])]
    # training_data = zip(inputs, ds)
    # bp.sgd(training_data, 1, 1, 0.5)
    # print(bp.weights)
    # print(bp.biases)
    # print("------------------------------")
    # res = bp.forward_cal(np.array([(1,), (2,)]))
    # print(res)
    # print("******************************")
# the results to compare
# [array([[-2.585, -2.582, -2.579],
#        [-5.17 , -5.264, -5.358]]), array([[-0.276, -0.76 ],
#        [-0.564, -1.64 ],
#        [-0.852, -2.52 ]])]
# [array([[-2.385],
#        [-2.382],
#        [-2.379]]), array([[-0.77],
#        [-1.2 ]])]



# test code
    # bp = dbp.DeepBP([2, 3, 2], True)
    # bp.weights = [np.array([(0.1, 0.2, 0.3), (0.2, 0.3, 0.4)]), np.array([(0.5, 1), (0.6, 1), (0.7, 1)])]
    # bp.biases = [np.array([(0.3,), (0.4,), (0.5,)]), np.array([(0.2,), (1,)])]
    # inputs = [np.array([(1,), (2,)])]
    # ds = [np.array([(0.5,), (0.2,)])]
    # training_data = zip(inputs, ds)
    # bp.forward_cal(inputs[0])
    # bp.cal_derivative(ds[0])




if __name__ == "__main__": main()










