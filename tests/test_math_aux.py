import pytest

import numpy as np
import numpy.testing as npt

# Enable local imports from parent directory
import sys
sys.path.append("..")

import math_aux as ma

@pytest.mark.parametrize("amplitude,spread", [
    (5,0.1),
    (0.1,5),
    (0,1)
])

def test_exponential_decay_1d_convolution_mx(amplitude, spread):
    #test_dx = 1 # This variable is not used in the manual calc
    #test_N = 3 # This variable is not used in the manual calc
    observed = ma.exponential_decay_1d_convolution_mx(1, 3,
        amplitude, spread)
    expected = amplitude/(2*spread) * np.array([
        [(np.exp(-np.abs(0-0)/spread)),
         (np.exp(-np.abs(0-1)/spread)),
         (np.exp(-np.abs(0-2)/spread))],
        [(np.exp(-np.abs(1-0)/spread)),
         (np.exp(-np.abs(1-1)/spread)),
         (np.exp(-np.abs(1-2)/spread))],
        [(np.exp(-np.abs(2-0)/spread)),
         (np.exp(-np.abs(2-1)/spread)),
         (np.exp(-np.abs(2-2)/spread))]
         ])
    # NOTE: this breaks if you move the first spread division to
    # each row, due to a difference in rounding.
    npt.assert_almost_equal(observed, expected)

def test_sholl_1d_connectivity_mx():
    #test_dx = 1
    #test_n_space = 3
    weights = [[1, 2], [3, 4]]
    spreads = np.array([[0.1, 0.2],[0.3, 0.4]])
    observed = ma.sholl_1d_connectivity_mx(1, 3,
        weights, spreads)
    expected = np.vstack((
        np.hstack((
            ma.exponential_decay_1d_convolution_mx(1, 3,
                weights[0][0], spreads[0][0]),
            ma.exponential_decay_1d_convolution_mx(1, 3,
                weights[0][1], spreads[0][1]))),
        np.hstack((
            ma.exponential_decay_1d_convolution_mx(1, 3,
                weights[1][0], spreads[1][0]),
            ma.exponential_decay_1d_convolution_mx(1, 3,
                weights[1][1], spreads[1][1])))))
    npt.assert_almost_equal(observed, expected)
