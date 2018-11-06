# -*- coding: utf-8 -*-

import random
import time
import numpy as np

def sigmoid_func(val):
    # Logistic function
    return 1.0 / (1.0 + np.exp(-val))

def sigmoid_func_prim(val):
    return sigmoid_func(val)*(1.0 - sigmoid_func(val))

def linear_func(val):
    return val

def linear_func_prim(val):
    return np.ones(val.shape)


class DeepBP(object):
    # init func
    def __init__(self, sizes, using_sigmoid=True):
        self.sizes = sizes
        self.layer_num = len(self.sizes)
        self.biases = None
        self.weights = None
        self.layers = []
        self.actived_layers = []
        self.using_sigmoid = using_sigmoid
        self.delta_t2 = 0.0
        self.__init_parameters()

    def active_func(self, val):
        if self.using_sigmoid:
            return sigmoid_func(val)
        return linear_func(val)

    def active_func_prim(self, val):
        if self.using_sigmoid:
            return sigmoid_func_prim(val)
        return linear_func_prim(val)

    def __init_parameters(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.sizes[0: -1], self.sizes[1:])]

    def forward_cal(self, val):
        self.layers.append(val)
        for weight, bias in zip(self.weights, self.biases):
            val = np.dot(np.transpose(weight), val) + bias
            self.layers.append(val)
            val = self.active_func(val)
            self.actived_layers.append(val)
        return val

    def cal_derivative(self, result):
        # prepare grad array for weigts and biases
        grad_ws = [np.zeros(w.shape) for w in self.weights]
        grad_bs = [np.zeros(b.shape) for b in self.biases]
        # biases derivative
        # δ_L = -(D - f(z_L))⊙f'(z_L)
        grad_bs[-1] = (self.actived_layers[-1] - result) * (self.active_func_prim(self.layers[-1]))
        # begin from (L - 1) layer
        for n in range(2, self.layer_num):
            grad_bs[-n] = np.dot(self.weights[-n + 1], grad_bs[-n + 1])
            grad_bs[-n] = np.multiply(grad_bs[-n], self.active_func_prim(self.layers[-n]))
        # weights derivative
        for n in range(0, self.layer_num - 1):
            if n != 0:
                a_val = self.actived_layers[n - 1]
            else:
                a_val = self.layers[0]
            grad_ws[n] = a_val * np.transpose(grad_bs[n])
        return grad_ws, grad_bs

    def batch_process(self, batch, eta):
        t1 = time.time()
        aveg_weights_derivative = None
        aveg_biases_derivative = None
        n = len(batch)
        delta_t = 0.0
        for x, y in batch:
            self.layers = []
            self.actived_layers = []
            t2 = time.time()
            self.forward_cal(x)
            weights_derivative, biases_derivative = self.cal_derivative(y)
            delta_t += (time.time() - t2)
            # accumnulate biases
            if aveg_weights_derivative is None:
                aveg_weights_derivative = np.array(weights_derivative)
            else:
                aveg_weights_derivative += np.array(weights_derivative)
            # accumulate weights
            if aveg_biases_derivative is None:
                aveg_biases_derivative = np.array(biases_derivative)
            else:
                aveg_biases_derivative += np.array(biases_derivative)
        aveg_weights_derivative *= (eta / n)
        aveg_biases_derivative *= (eta / n)
        for i in range(self.layer_num - 1):
            self.weights[i] -= aveg_weights_derivative[i]
            self.biases[i] -= aveg_biases_derivative[i]
        self.delta_t2 += delta_t

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
