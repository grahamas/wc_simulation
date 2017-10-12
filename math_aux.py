#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2 Aug 2017  Begin

import numpy as np
import itertools

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
def _exponential_decay_1d_convolution_mx(step, n_extent, amplitude, spread):
    """
        Calculates a weight matrix for the case of 1D WC with
        exponentially decaying spatial connectivity.
    """
    conv_mx = np.zeros((n_extent, n_extent))
    for i in range(n_extent):
        conv_mx[i, :] = amplitude *\
            np.exp(
                -np.abs(
                    step * (np.arange(n_extent)-i)
                    ) / spread
                ) *\
            step/(2*spread)
        # The division by 2*spread normalizes the beta,
        # to separate the scaling amplitude from the space constant spread
    return conv_mx
def sholl_1d_connectivity_mx(lattice, weights, spreads):
    n_pops = lattice.n_populations
    assert n_pops == len(weights)
    n_extent, step = lattice.n_space, lattice.space_step
    connectivity_mx = np.empty((n_pops*n_extent,n_pops*n_extent))
    for pop_pair in itertools.product(range(n_pops), range(n_pops)):
        to_pop = pop_pair[0]
        from_pop = pop_pair[1]
        connectivity_mx[stride(to_pop,n_extent),
            stride(from_pop,n_extent)] =\
                _exponential_decay_1d_convolution_mx(
                    step, n_extent, weights[to_pop][from_pop],
                    spreads[to_pop][from_pop]
                )

    return connectivity_mx
#endregion

def shift_op(arr, *, op, axis, shift=1):
    '''
        Computes binary operation op on pairs of elements separated by
        shift along axis of arr.
    '''
    n_dims = len(arr.shape)
    left = ([slice(None)] * n_dims)
    left[axis] = range(len(arr)-shift)
    right = ([slice(None)] * n_dims)
    right[axis] = range(shift,len(arr))
    return op(arr[left], arr[right])

def shift_subtract(arr, *, axis):
    return shift_op(arr, op=np.subtract, axis=axis)

def shift_multiply(arr, *, axis):
    return shift_op(arr, op=np.multiply, axis=axis)
