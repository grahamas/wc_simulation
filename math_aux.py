#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2 Aug 2017  Begin

import numpy as np

def stride(stride_num, stride_length):
    '''
        Return a slice representing a single stride (stride_num)
        implicitly through an array though the context is irrelevant,
        where the length of each stride is stride_length.
    '''
    return slice(stride_num*stride_length,(stride_num+1)*stride_length)

def awgn(arr, *, snr):
    """
        Add Gaussian white noise to arr, with signal-to-noise ratio of
        snr.
    """
    return arr + np.sqrt(np.power(10.0, -snr/10)) * np.random.randn(*arr.shape)

#region nonlinearities
def sigmoid(x, a, theta):
    """Standard two-parameter sigmoid."""
    return 1 / (1 + np.exp(-a * (x - theta)))

def sigmoid_norm(x, a, theta):
    """Sigmoid translated so that S(0) = 0"""
    return sigmoid(x,a,theta) - sigmoid(0,a,theta)

def sigmoid_norm_rectify(x,*, a, theta):
    """Sigmoid normed as above and rectified so S(x) = 0 for all x <= 0"""
    return np.maximum(0, sigmoid_norm(x, a, theta))

def sigmoid_rectify(x,*, a, theta):
    """Sigmoid rectified so S(x) = 0 for all x < 0"""
    return np.maximum(0, sigmoid(x,a,theta))

dct_nonlinearities = {
    'sigmoid_norm_rectify': sigmoid_norm_rectify
}
#endregion

#region calculate weight matrices
def exponential_decay_1d_convolution_mx(dx, Nx, amplitude, spread):
    """
        Calculates a weight matrix for the case of 1D WC with
        exponentially decaying spatial connectivity.
    """
    conv_mx = np.zeros((Nx, Nx))
    for i in range(Nx):
        conv_mx[i, :] = spread *
            np.exp(
                -np.abs(
                    dx * (np.arange(Nx)-i)
                    ) / spread
                ) *
            dx/(2*spread)
        # The division by 2*spread normalizes the beta,
        # to separate the scaling amplitude from the space constant spread
    return conv_mx
def sholl_1d_connectivity_mx(dx, n_space, weights, spreads):
    n_pops = len(weights)
    connectivity_mx = np.empty((n_pops*n_space,n_pops*n_space))
    for pop_pair in itertools.product(range(n_pops), range(n_pops)):
        to_pop = pop_pair[0]
        from_pop = pop_pair[1]
        connectivity_mx[math.stride(to_pop,n_space),
            math.stride(from_pop,n_space)] =
                exponential_decay_1d_convolution_mx(
                    dx, n_space, weights[to_pop][from_pop],
                    spreads[to_pop][from_pop]
                )

    return connectivity_mx
#endregion