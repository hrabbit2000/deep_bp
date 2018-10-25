# -*- coding: utf-8 -*-

import random
import numpy as np


class DeepBP(object):
    # init func
    def __init__(self, sizes, using_sigmoid = True):
        self.sizes = sizes
        self.layer_num = len(self.sizes)
        self.biases = None
        self.weights = None
        self.layers = []
        self.using_sigmoid = using_sigmoid

    def __sigmoid_func(self, z):
        # Logistic function
        return 1.0 / (1.0 + np.exp(-z))

    def __sigmoid_func_prim(self, z):
        return self.__sigmoid_func(z)*(1.0 - self.__sigmoid_func(z))

    def __square_func(self, z):
        return np.square(z)

    def __square_func_prim(self, z):
        return 2 * np.array(z)

    def active_func(self, z):
        if (self.using_sigmoid):
            return self.__sigmoid_func(z)
        else:
            return self.__square_func(z)

    def active_func_prim(self, z):
        if (self.using_sigmoid):
            return self.__sigmoid_func_prim(z)
        else:
            return self.__square_func_prim(z)

    def init_parameters(self):
        self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[0: -1], self.sizes[1:])]

    def forward_cal(self, input):
        self.layers.append(np.array(input))
        for w, b in zip(self.weights, self.biases):
            input = np.dot(w, input) + b[0]
            self.layers.append(input)
            input = self.active_func(input)
        #
        return input

    def cal_derivative(self, d):
        grad_array = [[] for i in range(self.layer_num)]
        derivative_weight_array = []
        derivative_biases_array = []
        # δ_L = -(D - f(z_L))⊙f'(z_L)
        grad_final = np.multiply(self.active_func(self.layers[-1:][0]) - d, 
                                 self.active_func_prim(self.layers[-1:][0]))
        grad_array[self.layer_num - 1] = grad_final
        # begin from (L - 1) layer
        for n in range(self.layer_num - 2, 0, -1):
            grad_n = np.dot(np.transpose(self.weights[(n - 1) + 1]), grad_array[n + 1])
            grad_n = np.multiply(grad_n, self.active_func_prim(self.layers[n]))
            grad_array[n] = grad_n
        # biases derivative
        derivative_biases_array = grad_array[1:]
        # weights derivative
        for n in range(1, self.layer_num):
            itemT = np.transpose(np.matrix(grad_array[n]))
            a_val = self.active_func(self.layers[n - 1])
            if (1 == n):
                a_val = self.layers[0]
            derivative_weight_array.append(itemT * a_val)
        return derivative_weight_array, derivative_biases_array

    def batch_process(self, inputs, ds, eta):
        aveg_weights_derivative = None
        aveg_biases_derivative = None
        self.layers = []
        n = len(inputs)
        for i in range(n):
            self.forward_cal(inputs[i])
            weights_derivative, biases_derivative = self.cal_derivative(ds[i])
            # accumnulate biases
            if None == aveg_weights_derivative:
                aveg_weights_derivative = np.array(weights_derivative)
            else:
                aveg_weights_derivative += np.array(weights_derivative)
            # accumulate weights
            if None == aveg_biases_derivative:
                aveg_biases_derivative = np.array(biases_derivative)
            else:
                aveg_biases_derivative += np.array(biases_derivative)
        aveg_weights_derivative *= (eta / n)
        aveg_biases_derivative *= (eta / n)
        aveg_weights_derivative = self.__convert_matrixs_to_arrays(aveg_weights_derivative)
        for i in range(self.layer_num - 1):
            self.weights[i] -= aveg_weights_derivative[i]
            self.biases[i] -= aveg_biases_derivative[i]

    def sgd(self, zip_data, epochs, mini_batch, eta, test_data=None):
        for i in range(epochs):
            random.shuffle(zip_data)
            mini_batches = [zip_data[k : k + mini_batch] 
                            for k in range(0, len(zip_data), mini_batch)]
            for mini_batche in mini_batches:
                for x, y in mini_batche:
                    self.batch_process(x, y, eta)

    def __convert_matrixs_to_arrays(self, matrixs):
        arrays = []
        for m in matrixs:
            arrays.append(m.getA())
        return arrays




