#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import os
sys.path.append(os.path.abspath("./"))

import mnist_loader
import network
import numpy as np

print "开始训练，较耗时，请稍等。。。"

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# 784 个输入神经元，一层隐藏层，包含 30 个神经元，输出层包含 10 个神经元
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)
# Epoch 0: 9038 / 10000
# Epoch 1: 9178 / 10000
# Epoch 2: 9231 / 10000
# ...
# Epoch 27: 9483 / 10000
# Epoch 28: 9485 / 10000
# Epoch 29: 9477 / 10000



# bp = network.Network([2, 3, 2])
# bp.weights = [np.array([(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]), np.array([(0.5, 0.6, 0.7), (1.0, 1.0, 1.0)])]
# bp.biases = [np.array([(0.3,), (0.4,), (0.5,)]), np.array([(0.2,), (1.0,)])]
# inputs = [np.array([(1,), (2,)])]
# ds = [np.array([(0.5,), (0.2,)])]
# training_data = zip(inputs, ds)
# bp.SGD(training_data, 10, 1, 0.5)
# print(bp.weights)
# print(bp.biases)
# print("------------------------------")
# res = bp.feedforward(np.array([(1,), (2,)]))
# print(res)
# print("******************************")



