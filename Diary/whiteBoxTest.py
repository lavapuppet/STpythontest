import numpy as np
import unittest
import random as rand

from numpy import array, single, double, csingle, cdouble, dot, identity,empty
from numpy import multiply, atleast_2d, inf, asarray, matrix
import linpy 



class WhiteBox(unittest.TestCase):
    def setUp(self):
        self.array_2_2_identity = np.identity(2)
        self.array_1 = np.array([[1,2],[3,4]])
        self.array_2 = np.array([[5,6],[7,8]])
        self.array_3 = np.array([[9,10,11],[12,13,14],[15,16,17]])
        self.array_4 = np.array([[6,5],[5,3],[12,15]])
        #self.array_multi_5 = np.rand.rand(3,2)

    
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
        actual = linpy.multi_dot([self.sint,self.array_1, 
            self.array_2,self.sfloat])
        expected = np.array([6190.8])
        self.assertAlmostEqual(actual, expected)
        

if __name__ == '__main__':
    unittest.main()
