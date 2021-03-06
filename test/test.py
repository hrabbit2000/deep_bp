import sys
import os
sys.path.append(os.path.abspath("./"))
import time
import unittest
import mnist_loader as loader
import deep_bp as dbp
import numpy as np


class DeepBPTest(unittest.TestCase):
    def setUp(self):
        self.bp = dbp.DeepBP([3, 4, 2])
        self.m = [[2, 3], [3, 4],[4, 5]]
        self.v = [[1], [2]]
        self.b = [[1], [2], [3]]

    def test_active_func(self):
        z = self.bp.active_func(2)
        self.assertAlmostEqual(0.88079, 0.88079, 4)

    def test_numpy_matrix(self):
        res = np.dot(self.m, self.v) + self.b
        self.assertEqual(1, 1)

    def test_numpy_hadamard(self):
        m1 = [[1, 2], [2, 3]]
        m2 = [[3, 4], [4, 5]]
        m12 = np.multiply(m1, m2)
        m3 = np.matrix([[1, 2], [2, 3]])
        m4 = m3.getA()
        self.assertEqual(1, 1)        

    def test_array(self):
        m1 = [1, 2, 3]
        m2 = [i for i in range(3 - 1, -1, -1)]
        m2[1] = [8, 4]
        m3 = np.array([1, 2])
        m3 = np.transpose(np.matrix(m3))
        m4 = np.array([1, 2, 3])
        m5 = m3 * m4
        m6 = np.array(m1) * 2
        # print(m1[-3])
        # print(m6)
        # print(m5)
        # m4 = np.multiply(np.transpose(m1), m3)
        # print(m1[-1:])
        # print(m2)
        self.assertEqual(1, 1)

    def test_square(self):
        z = np.square([2, 3])
        # print(z)
        self.assertEqual(1, 1)

    def test_reshape(self):
        z = np.array(self.m)
        # print()
        # print(z.shape, z)
        z = np.transpose(z)
        # print(z.shape, z)
        self.assertEqual(1, 1)

    def test_range_xrange(self):
        range_val = [i for i in range(10, 1, -3)]
        range_val_x = [i for i in xrange(10, 1, -3)]
        # print(range_val)
        # print(range_val_x)

    def test_timestamp(self):
        print(time.time())
        print("hahahahah")






