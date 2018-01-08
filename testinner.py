import numpy as np
import unittest

class TestLinAlg(unittest.TestCase):
    def setUp(self):
        
        self.array_a = np.array([1, 2, 3])
        self.array_b = np.array([0, 1, 0])

        self.array_a_float = np.array([1.0, 2.0, 4.5])
        self.array_b_float = np.array([3.0, 3.5, 2.5])

        """Inner Product Wolfram"""
        self.vector_u = np.array([1,2,3])
        self.vector_v = np.array([1,2,1])
        self.vector_w = np.array([4,5,6])
        self.scalar = 5
        self.vector_zero = np.array([0, 0, 0])

        """
            Testing numpy.linalg.inner() function
        """
        
    def test_inner_simple(self):
        actual = np.inner(self.array_a, self.array_b)
        expected = 2
        self.assertTrue(actual == expected)

    def test_inner_zero(self):
        actual = np.inner(self.array_a, [0, 0, 0])
        expected = 0
        self.assertTrue(actual == expected)

    def test_inner_float(self):
        actual = np.inner(self.array_a_float, self.array_b_float)
        expected = 21.25
        self.assertTrue(actual == expected)

    """http://mathworld.wolfram.com/InnerProduct.html"""
    def test_inner_prop1(self): 
        actual = np.inner(np.add(self.vector_u,self.vector_v), self.vector_w)
        expected = np.add(np.inner(self.vector_u,self.vector_w),
                          np.inner(self.vector_v,self.vector_w))
        self.assertTrue(actual == expected)

    def test_inner_prop2(self):
        actual = np.inner(self.scalar * self.vector_v, self.vector_w)
        expected = self.scalar * np.inner(self.vector_v, self.vector_w)
        self.assertTrue(actual == expected)

    def test_inner_prop3(self):
        actual = np.inner(self.vector_v, self.vector_w)
        expected = np.inner(self.vector_w, self.vector_v)
        self.assertTrue(actual == expected)

    def test_inner_prop4(self):
        actual = np.inner(self.vector_zero, self.vector_zero)
        expected = 0
        self.assertTrue(actual == expected)
        
    def test_inner_raises(self):
        with self.assertRaises(ValueError):
            actual = np.inner([2, 2, 3], [2, 1])


if __name__ == '__main__':
    unittest.main()
