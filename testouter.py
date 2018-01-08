import numpy as np
import unittest

class TestLinAlg(unittest.TestCase):
    def setUp(self):
        
        self.vector_a = np.array(['a', 'b', 'c'], dtype=object)
        self.vector_b = np.array([1, 2, 3])
        self.vector_c = np.array([1, 2])
        self.vector_zero = np.array([0, 0, 0], dtype=object)

        self.vector_com_a = np.array([1+2j, 2+3j]) 
        self.vector_com_b = np.array([1+2j, 2+3j])
     
        """
            Testing numpy.linalg.outer() function
        """
        
    def test_outer_simple(self):
        actual = np.outer(self.vector_a, self.vector_b)
        expected = [['a', 'aa', 'aaa'],['b', 'bb', 'bbb'],['c', 'cc', 'ccc']]
        self.assertTrue((actual == expected).all())

    def test_outer_zero(self):
        actual = np.outer(self.vector_b, self.vector_zero)
        expected = np.zeros((3, 3))
        self.assertTrue((actual == expected).all())

    def test_outer_complex(self):
        actual = np.outer(self.vector_com_a, self.vector_com_b)
        expected = np.array([[-3. +4.j, -4. +7.j],[-4. +7.j, -5.+12.j]])
        self.assertTrue((actual == expected).all())

    def test_outer_dimensions(self):
        actual = np.outer(self.vector_a, self.vector_c)
        expected = [['a', 'aa'],['b', 'bb'],['c', 'cc']]
        self.assertTrue((actual == expected).all())

if __name__ == '__main__':
    unittest.main()
