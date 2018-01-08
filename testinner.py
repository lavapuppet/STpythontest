import numpy as np
import unittest

class TestLinAlg(unittest.TestCase):
    def setUp(self):
        
        self.array_a = np.array([1, 2, 3])
        self.array_b = np.array([0, 1, 0])

        self.array_a_float = np.array([1.0, 4.5])
        self.array_b_float = np.array([3.0, 2.5])

        """
            Testing numpy.linalg.vdot() function
        """
        
    def setupForComplex(self):
        self.complex_a = np.array([1+2j,3+4j])   
        self.complex_b = np.array([5+6j,7+8j])
        self.complex_c = np.array([-5-6j,-7-8j])

    def test_vdot_square(self):
        actual = np.vdot(self.array_a, self.array_a)
        expected = 78

        self.assertTrue(actual == expected)

    def test_vdot_matrix(self):
        actual = np.vdot(self.array_a, self.array_a)
        expected = 78

        self.assertTrue(actual == expected)
        
    def test_vdot_normal(self):
        self.setupForComplex()

        actual = np.vdot(self.complex_a, self.complex_b)
        expected = 70-8j

        self.assertTrue(actual == expected)

    def test_vdot_com(self):
        self.setupForComplex()
        
        actual = np.vdot(self.array_a, self.array_b)
        expected = np.vdot(self.array_b, self.array_a)

        self.assertTrue(actual == expected)

    def test_vdot_negative(self):
        self.setupForComplex()

        actual = np.vdot(self.complex_c, self.complex_a)
        expected = -70-8j

        self.assertTrue(actual == expected)


    def test_vdot_float(self):
        actual = np.vdot(self.array_a_float, self.array_b_float)
        expected = 14.25

        self.assertTrue(actual == expected)

    def test_vdot_empty(self):
        actual = np.vdot([],[])

        self.assertFalse(actual)

    

    
if __name__ == '__main__':
    unittest.main()
