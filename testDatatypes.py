
import os
import sys
import itertools
import traceback
import warnings

import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity
from numpy import multiply, atleast_2d, inf, asarray, matrix
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, run_module_suite,
    dec, SkipTest, suppress_warnings
)

import unittest


class LinalgCase(object):
    def __init__(self, name, a, b, tags=set()):
        """
        A bundle of arguments to be passed to a test case, with an identifying
        name, the operands a and b, and a set of tags to filter the tests
        """
        assert_(isinstance(name, str))
        self.name = name
        self.a = a
        self.b = b
        self.tags = frozenset(tags)  # prevent shared tags




CASES = []

CASES += [LinalgCase("single",
                     array([1., 1., 1.], dtype=single),
                     array([1., 1., 1.], dtype=single)),
          LinalgCase("double",
                     array([[1., 2.], [3., 4.]], dtype=double),
                     array([2., 1.], dtype=double)),
          ]



class TestDatatypes(unittest.TestCase):

    def test_temp1(self):
        self.assertEqual(5, 5)

    def test_temp2(self):
        ggg = CASES[0]
        self.assertEqual(np.dot(ggg.a, ggg.b), 3)

    def test_invTypes(self):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=single)
        assert_equal(linalg.solve(x, x).dtype, single)

    def test_EigTypes(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            assert_equal(linalg.eigvals(x).dtype, dtype)
        for dtype in [single, double, csingle, cdouble]:
            yield check, dtype

if __name__ == '__main__':
    unittest.main()
