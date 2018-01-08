import numpy as np
import unittest

class TestLinAlg(unittest.TestCase):
    def setUp(self):
        
        self.vector_a = np.array([[1, 0],[0, 1]])
        self.vector_b = np.array([[4, 1],[2, 2]])

        self.vector_2d = np.array([[1, 0],[0, 1]])
        self.vector_1d = np.array([1,2])

        """http://mathworld.wolfram.com/MatrixMultiplication.html"""
        self.matrix_a = np.array([[1, 0],[0, 1]])
        self.matrix_b = np.array([[4, 1],[2, 2]])
        self.matrix_c = np.array([[3, 2],[2, 3]])

        self.matrix_diag_a = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.matrix_diag_b = np.array([[2,0,0],[0,2,0],[0,0,2]])       
     
        """
            Testing numpy.linalg.matmul() function
        """
        
    def test_matmul_simple(self):
        actual = np.matmul(self.vector_a, self.vector_b)
        expected = np.array([[4,1],[2,2]])
        self.assertTrue((actual == expected).all())

    def test_matmul_mix(self):
        actual = np.matmul(self.vector_2d, self.vector_1d)
        expected = np.array([1,2])
        self.assertTrue((actual == expected).all())

    def test_matmul_distributivity(self):
        actual = np.matmul(self.matrix_a, np.add(self.matrix_b, self.matrix_c))
        expected = np.add(np.matmul(self.matrix_a, self.matrix_b),np.matmul(self.matrix_a, self.matrix_c))
        self.assertTrue((actual == expected).all())

    def test_matmul_diag_commutative(self):
        actual = np.matmul(self.matrix_diag_a, self.matrix_diag_b)
        expected = np.matmul(self.matrix_diag_b, self.matrix_diag_a)
        self.assertTrue((actual == expected).all())
        
    def test_matmul_raises(self):
        with self.assertRaises(ValueError):
            actual = np.matmul([2, 2, 3], [2, 1])

    def test_matmul_scalar_raises(self):
        with self.assertRaises(ValueError):
            actual = np.matmul([2, 2, 3], 2)

if __name__ == '__main__':
    unittest.main()
