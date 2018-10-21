#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np


class DeepBP(object):
    # init func
    def __init__(self, sizes):
        self.sizes = sizes
        self.layer_num = None
        self.biases = None
        self.weights = None
        self.layers = []

    def active_func(self, z):
        # Logistic function
        return 1.0 / (1.0 + np.exp(-z))

    def active_func_prim(self, z):
        return self.active_func(z)*(1.0 - self.active_func(z))

    def init_parameters(self):
        self.layer_num = len(self.sizes)
        self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[0: -1], self.sizes[1:])]

    def forward_cal(self, inputs):
        self.layers.append(np.array(inputs))
        for w, b in zip(self.weights, self.biases):
            inputs = np.dot(w, inputs) + b[0]
            self.layers.append(inputs)
            inputs = self.active_func(inputs)

    def cal_grad(self, d):
        grad_array = []
        # δ_L = -(D - f(z_L))⊙f'(z_L)
        grad_final = np.multiply(self.active_func(self.layers[-1:][0]) - d, 
                                 self.active_func_prim(self.layers[-1:][0]))
        grad_array.append(grad_final);
        #
        for i in range(len(self.weights)):
            z = np.dot(np.transpose(self.weights[len(self.weights) -i - 1]), grad_array[i])
            z = np.multiply(z, self.active_func_prim(self.layers[len(self.layers) - i - 1 - 1]))
            grad_array.append(z)



