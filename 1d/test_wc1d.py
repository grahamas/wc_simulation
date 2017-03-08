import unittest
import wc1d as wc
import numpy as np
import numpy.testing as npt


class TestParamMethods(unittest.TestCase):

    def test_weight_mx(self):
        test_N = 3
        observed = wc.make_weight_mx(1, test_N, 1)
        expected = np.array([
            [(np.exp(-np.abs(0-0))/2)/2,
             np.exp(-np.abs(0-1))/2,
             (np.exp(-np.abs(0-2))/2)/2],
            [(np.exp(-np.abs(1-0))/2)/2,
             np.exp(-np.abs(1-1))/2,
             (np.exp(-np.abs(1-2))/2)/2],
            [(np.exp(-np.abs(2-0))/2)/2,
             np.exp(-np.abs(2-1))/2,
             (np.exp(-np.abs(2-2))/2)/2]])
        npt.assert_array_equal(observed, expected)

if __name__ == '__main__':
    unittest.main()
