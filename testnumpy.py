import numpy as np
import unittest

class TestLinAlg(unittest.TestCase):
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
        self.array_per_1com = [2.0 , 1.0, 4.0]
        self.array_per_2com = [1.0 , -1.0, -0.25]

        """ Complex nubmers 
        self.array_per_1 = [2 +4j , 1+3j, 4.0]
        self.array_per_2 = [1.0+4j, -1.0+3j, -0.25] """




        """
            Testing numpy.linalg.dot() function
        """
    
    def test_dot_raises(self):
        with self.assertRaises(ValueError):
            actual = np.dot([2, 2, 3], [2, 1])

    def test_dot_corner(self):
        actual = np.dot([], [])
        expected = False
        self.assertEqual(actual, expected);

    def test_dot_corner2(self):
        with self.assertRaises(ValueError):
            actual = np.dot([], [1, 2])
    
    def test_dot_iden(self):
        actual = np.dot(self.array_1, self.array_2)
        expected = [[4, 1], [2, 2]]
        self.assertTrue((actual == expected).all())

    def test_dot_zero(self):
        actual = np.dot(self.array_zero, self.array_2)
        expected = 0
        self.assertTrue((actual == expected).all())
        
    def test_dot_com(self):
        actual = np.dot(self.array_com_1, self.array_com_2)
        expected = np.dot(self.array_com_2, self.array_com_1)
        self.assertTrue((actual == expected).all())

    def test_dot_square(self):
        actual = np.dot(self.array_2, self.array_2)
        expected = [[18, 6], [12, 6]]
        self.assertTrue((actual == expected).all())

    #def test_dot_perpendicular(self):        
        #actual = np.dot(self.array_per_1, self.array_per_2)
        #expected = 0
        #self.assertTrue((actual == expected).all())
        
        """
            Testing numpy.linalg.vdot() function
        """
    def setupForComplex(self):
        self.complex_a = np.array([1+2j,3+4j])   
        self.complex_b = np.array([5+6j,7+8j])

        self.complex_a1 = np.array([[1, 4], [5, 6]])
        self.complex_b1 = np.array([[4, 1], [2, 2]])

    def test_vdot_normal(self):
        self.setupForComplex()

        actual = np.vdot(self.complex_a, self.complex_b)
        expected = 70-8j

        self.assertTrue((actual == expected).all())

    def test_vdot_com(self):
        self.setupForComplex()
        
        actual = np.vdot(self.complex_a1, self.complex_b1)
        expected = np.vdot(self.complex_b1, self.complex_a1)

        self.assertTrue((actual == expected).all())

    def test_vdot_zero(self):
        self.setupForComplex()
        
        actual = np.vdot(self.complex_a1, 0)
        expected = 0

        self.assertTrue((actual == expected).all())

    def test_vdot_square(self):
        actual = np.vdot(self.array_2, self.array_2)
        expected = 25
        self.assertTrue((actual == expected).all())

    
    
if __name__ == '__main__':
    unittest.main()
