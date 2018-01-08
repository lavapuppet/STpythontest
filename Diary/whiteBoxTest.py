import numpy as np
import unittest
import random as rand

from numpy import array, single, double, csingle, cdouble, dot, identity,empty
from numpy import multiply, atleast_2d, inf, asarray, matrix
import linpy 
#from linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
#from linalg import _multi_dot_matrix_chain_order
#from numpy.testing import (
#    assert_, assert_equal, assert_raises, assert_array_equal,
#    assert_almost_equal, assert_allclose, run_module_suite,
#    dec, SkipTest)



class WhiteBox(unittest.TestCase):
    def setUp(self):
        self.array_2_2_identity = np.identity(2)
        self.array_1 = np.array([[1,2],[3,4]])
        self.array_2 = np.array([[5,6],[7,8]])
        self.array_3 = np.array([[9,10,11],[12,13,14],[15,16,17]])
        self.array_4 = np.array([[6,5],[5,3],[12,15]])
        #self.array_multi_5 = np.rand.rand(3,2)

        """ Empty Test [1](http://gettingsharper.de/2011/11/30/vector-fun-dot-product/)"""
        self.array_empty = np.empty([2,2])

        """Linear Test [1]"""
        self.array_com_3 = [12.4, -1.7, 3.15]
        self.scalar = 0.22
        self.sint = np.array([7,7])
        self.sfloat = np.array([6.6,6.6])

        """ Basic test no need to test with floats/ints/complex as this function doesn't differentiate """
        """ Perpendicular Test [1]"""
        self.array_per_1 = [2.0 , 1.0, 4.0]
        self.array_per_2 = [1.0 , -1.0, -0.25]

    """
        Testing numpy.linalg.dot() function
    """
    
    def test_multi_dot_raises(self):
        with self.assertRaises(ValueError):
            actual = linpy.multi_dot([self.array_1])
        
#        try:
 #           self.assertFalse(True)
  #      except ValueError:
   #         pass
        
    #test 2
    def test_multi_two(self):
        actual = linpy.multi_dot([self.array_1, self.array_2])
        expected1 = np.array([[19, 22], [43, 50]])
        self.assertTrue((actual == expected1).all())

    def test_multi_ndim_01(self):
        actual = linpy.multi_dot([self.sint,self.array_1, self.array_2])
        expected = np.array([434,504])
        self.assertTrue((actual == expected).all())
        
    def test_multi_ndim_00(self):
        actual = linpy.multi_dot([self.array_4,self.array_1, self.array_2])
        expected = np.array([[329,382],[224,260],[873,1014]])
        self.assertTrue((actual == expected).all())

    def test_multi_ndim_end1(self):
        actual = linpy.multi_dot([self.array_1, self.array_2,self.sint])
        expected = np.array([287,651])
        self.assertTrue((actual == expected).all())
    
    
    def test_multi_ndim_11(self):
        actual = linpy.multi_dot([self.sint,self.array_1, self.array_2,self.sfloat])
        expected = np.array([6190.8])
        self.assertAlmostEqual(actual, expected)
        

if __name__ == '__main__':
    unittest.main()
