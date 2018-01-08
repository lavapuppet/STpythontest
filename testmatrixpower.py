import numpy as np
from numpy import linalg as LA
import unittest

class TestLinAlg(unittest.TestCase):
    def setUp(self):
        self.matrix_a = np.array([[4, 1, 2],[2, 2, 1],[3, 2, 2]])
        self.matrix_b = np.matrix([[0, 1],[-1, 0]])
     
        """
            Testing numpy.linalg.matrix_power() function
        """
        
    def test_matrixpower_simple(self):
        actual = LA.matrix_power(self.matrix_a, 3)
        expected = np.array([[155, 70, 84],[100, 47, 54],[146, 68, 79]])
        self.assertTrue((actual == expected).all())

    def test_matrixpower_negative(self):
        actual = LA.matrix_power(self.matrix_b, -3)
        expected = np.array([[0., 1.],[-1., 0.]])
        self.assertTrue((actual == expected).all())

    def test_matrixpower_identity(self):
        actual = LA.matrix_power(self.matrix_a, 0)
        expected = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        self.assertTrue((actual == expected).all())
    
if __name__ == '__main__':
    unittest.main()
