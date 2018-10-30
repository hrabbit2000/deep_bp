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
        self.__init_parameters()

    def __sigmoid_func(self, z):
        # Logistic function
        return 1.0 / (1.0 + np.exp(-z))

    def __sigmoid_func_prim(self, z):
        return self.__sigmoid_func(z)*(1.0 - self.__sigmoid_func(z))

    def __linear_func(self, z):
        return z

    def __linear_func_prim(self, z):
        return np.ones(z.shape)

    def active_func(self, z):
        if (self.using_sigmoid):
            return self.__sigmoid_func(z)
        else:
            return self.__linear_func(z)

    def active_func_prim(self, z):
        if (self.using_sigmoid):
            return self.__sigmoid_func_prim(z)
        else:
            return self.__linear_func_prim(z)

    def __init_parameters(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.sizes[0: -1], self.sizes[1:])]

    def forward_cal(self, input):
        z = input
        self.layers.append(np.array(z))
        for w, b in zip(self.weights, self.biases):
            z = np.dot(np.transpose(w), z) + b
            self.layers.append(z)
            z = self.active_func(z)
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
            grad_n = np.dot(self.weights[(n - 1) + 1], grad_array[n + 1])
            grad_n = np.multiply(grad_n, self.active_func_prim(self.layers[n]))
            grad_array[n] = grad_n
        # biases derivative
        derivative_biases_array = grad_array[1:]
        # weights derivative
        for n in range(1, self.layer_num):
            itemT = grad_array[n]
            a_val = self.active_func(self.layers[n - 1])
            if (1 == n):
                a_val = self.layers[0]
            derivative_weight_array.append(a_val * np.transpose(itemT))
        return derivative_weight_array, derivative_biases_array

    def batch_process(self, batch, eta):
        aveg_weights_derivative = None
        aveg_biases_derivative = None
        self.layers = []
        n = len(batch)
        for x, y in batch:
            self.forward_cal(x)
            weights_derivative, biases_derivative = self.cal_derivative(y)
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
        for i in range(self.layer_num - 1):
            self.weights[i] -= aveg_weights_derivative[i]
            self.biases[i] -= aveg_biases_derivative[i]

    def sgd(self, zip_data, epochs, mini_batch, eta, test_data=None):
        for j in range(epochs):
            random.shuffle(zip_data)
            mini_batches = [zip_data[k : k + mini_batch] 
                            for k in range(0, len(zip_data) - mini_batch + 1, mini_batch)]
            for mini_batche in mini_batches:
                self.batch_process(mini_batche, eta)
            #
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), len(test_data))
            else:
                print "Epoch {0} complete".format(j)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.forward_cal(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)



