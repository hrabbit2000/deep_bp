import sys
import os
sys.path.append(os.path.abspath("./"))
import unittest
import deep_bp as dbp
import numpy as np


class DeepBPTest(unittest.TestCase):
    def setUp(self):
        self.bp = dbp.DeepBP([3, 4, 2])
        self.m = [[2, 3], [3, 4],[4, 5]]
        self.v = [[1], [2]]
        self.b = [[1], [2], [3]]

    def test_active_func(self):
        self.assertAlmostEqual(self.bp.active_func(2), 0.88079, 4)

    def test_numpy_matrix(self):
        res = np.dot(self.m, self.v) + self.b
        self.assertEqual(1, 1)

    def test_numpy_hadamard(self):
        m1 = [[1, 2], [2, 3]]
        m2 = [[3, 4], [4, 5]]
        m12 = np.multiply(m1, m2)
        self.assertEqual(1, 1)        

    def test_array(self):
        m1 = [1, 2, 3, 4, 5]
        print(m1[-1:])
        self.assertEqual(1, 1)        

