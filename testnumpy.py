import numpy as np
import unittest

class TestLinAlg(unittest.TestCase):
    """
    NP.DOT()
    testing55
    """

    def setUp(self):
        """ Basic Identity test """
        self.array_1 = [[1, 0], [0, 1]]
        self.array_2 = [[4, 1], [2, 2]]

        """ Zero Test [1](http://gettingsharper.de/2011/11/30/vector-fun-dot-product/)"""
        self.array_zero = [0, 0]

        """ Commutative Test [1]"""
        self.array_com_1 = [-3.22 , 2.25, -0.13]
        self.array_com_2 = [0.0 , -6.7, 10.0]

        """Linear Test [1]"""
        self.array_com_3 = [12.4, -1.7, 3.15]
        self.scalar = 0.22

        """ Square Test [1]"""

        """ Perpendicular Test [1]"""
        self.array_per_1 = [2.0 , 1.0, 4.0]
        self.array_per_2 = [1.0 , -1.0, -0.25]
        
        
    def test_dot_iden(self):
        result_array = np.dot(self.array_1, self.array_2)
        self.assertTrue((result_array == [[4, 1], [2, 2]]).all())

    def test_dot_zero(self):
        result_zero = np.dot(self.array_zero, self.array_2)
        self.assertTrue((result_zero == 0).all())
        
    def test_dot_com(self):
        result_array_left = np.dot(self.array_com_1, self.array_com_2)
        result_array_right = np.dot(self.array_com_2, self.array_com_1)
        self.assertTrue((result_array_left == result_array_right).all())

    def test_dot_lin(self):
        result_array_lin_left = np.dot(self.array_com_1,(np.dot(self.scalar, self.array_com_2)+ self.array_com_3))
        print(result_array_lin_left)
        result_array_lin_right = self.scalar * (np.dot(self.array_com_1, self.array_com_2) + (np.dot(self.array_com_1 ,self.array_com_3)))
        print(result_array_lin_right)
        self.assertTrue((result_array_lin_left == result_array_lin_right).all())

    def test_dot_square(self):
        result_array_square = np.dot(self.array_2, self.array_2)
        self.assertTrue((result_array_square == [[18, 6], [12, 6]]).all())

    def test_dot_perpendicular(self):
        result_array_perpendicular = np.dot(self.array_per_1, self.array_per_2)
        self.assertTrue((result_array_perpendicular == 0).all())
    

if __name__ == '__main__':
    unittest.main()
