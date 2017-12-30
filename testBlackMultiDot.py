

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

# code to be white box tested ********************************************************************
# # optimization only makes sense for len(arrays) > 2
#     if n < 2:
#         raise ValueError("Expecting at least two arrays.")
#     elif n == 2:
#         return dot(arrays[0], arrays[1])
#
# # save original ndim to reshape the result array into the proper form later
#     ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
#
#     # Explicitly convert vectors to 2D arrays to keep the logic of the internal
#     # _multi_dot_* functions as simple as possible.
#     if arrays[0].ndim == 1:
#         arrays[0] = atleast_2d(arrays[0])
#     if arrays[-1].ndim == 1:
#         arrays[-1] = atleast_2d(arrays[-1]).T
#     _assertRank2(*arrays)
#
# # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
#     if n == 3:
#         result = _multi_dot_three(arrays[0], arrays[1], arrays[2])
#     else:
#         order = _multi_dot_matrix_chain_order(arrays)
#         result = _multi_dot(arrays, order, 0, n - 1)
#
#     # return proper shape
#     if ndim_first == 1 and ndim_last == 1:
#         return result[0, 0]  # scalar
#     elif ndim_first == 1 or ndim_last == 1:
#         return result.ravel()  # 1-D
#     else:
#         return result
#
    #################################################################################################



CASES = []

dimension1 = np.random.randint(1, 12)
dimension2 = np.random.randint(1, 12)

CASES += [np.random.random((dimension1, dimension2)), # 0
                np.random.random((dimension2, dimension1)),
                np.random.random((dimension1, dimension2)),
                np.random.random((dimension2, dimension1)),
                np.random.random(dimension1),
                np.random.random(dimension1), # 5
                np.random.random(dimension1)]


class TestMultiDot(unittest.TestCase):

    def test_two_inputs_vectors(self):
        A = CASES[4]
        B = CASES[5]

        assert_almost_equal(multi_dot([A, B]), A.dot(B))
        assert_almost_equal(multi_dot([A, B]), np.dot(A, B))


    def test_three_inputs_vectors(self):

        A = CASES[4]
        B = CASES[5]
        C = CASES[2]

        # running multi-dot on three equal size vectors should result in a multi_dot error
        try:
            assert_almost_equal(multi_dot([A, B, C]), np.dot(np.dot(A, B), C))
        except Exception:
            pass


    def test_three_inputs_matrices(self):
        A = CASES[0]
        B = CASES[1]
        C = CASES[2]

        assert_almost_equal(multi_dot([A, B, C]), A.dot(B).dot(C))
        assert_almost_equal(multi_dot([A, B, C]), np.dot(A, np.dot(B, C)))

        #when the inputs are in the wrong order
        try:
            assert_almost_equal(multi_dot([A, C, B]), np.dot(np.dot(A, C), C))
        except Exception:
            pass


    def test_four_inputs_matrices(self):
        A = CASES[0]
        B = CASES[1]
        C = CASES[2]
        D = CASES[3]

        assert_almost_equal(multi_dot([A, B, C, D]), A.dot(B).dot(C).dot(D))

        # when the inputs are in the wrong order
        try:
            assert_almost_equal(multi_dot([A, C, B]), np.dot(np.dot(A, C), C))
        except Exception:
            pass

    def test_vector_as_first_argument(self):
        # The first argument can be 1-D
        A1d = np.random.random(2)  # 1-D
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D = np.random.random((2, 2))

        # the result should be 1-D
        assert_equal(multi_dot([A1d, B, C, D]).shape, (2,))

    def test_vector_as_last_argument(self):
        # The last argument can be 1-D
        A = np.random.random((6, 2))
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D1d = np.random.random(2)  # 1-D

        # the result should be 1-D
        assert_equal(multi_dot([A, B, C, D1d]).shape, (6,))

    def test_vector_as_first_and_last_argument(self):
        # The first and last arguments can be 1-D
        A1d = np.random.random(2)  # 1-D
        B = np.random.random((2, 6))
        C = np.random.random((6, 2))
        D1d = np.random.random(2)  # 1-D

        # the result should be a scalar
        assert_equal(multi_dot([A1d, B, C, D1d]).shape, ())




if __name__ == '__main__':
    unittest.main()
