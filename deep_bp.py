
import numpy as np


class DeepBP(object):
    # init func
    def __init__(self, sizes):
        self.sizes = sizes
        self.layer_num = None
        self.biases = None
        self.weights = None

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
        for w, b in zip(self.weights, self.biases):
            inputs = self.active_func(np.dot(w, inputs) + b[0])
        return inputs

    def cal_grad(self, output, real_val):
        pass




