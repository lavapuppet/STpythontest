import Numpy as np
import unittest

class TestLinAlg(unittest.TestCase):
    """

    """

    def setUp(self):
        self.array_1 = [[1, 0], [0, 1]]
        self.array_2 = [[4, 1], [2, 2]]

    def test_dot(self):
        (result_array) = np.dot(self.array_1, self.array_2)
        self.assertEqual(result_array , [[4, 1], [2, 2]])


if __name__ == '__main__':
    unittest.main()
