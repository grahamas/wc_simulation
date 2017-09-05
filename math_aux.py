#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2 Aug 2017  Begin

import numpy as np

def central_window(total_len, window_width):
    """
        Calculate indices for a region of width window_width roughly in the
        center of an array of length total_len.
    """
    median_dx = total_len // 2
    half_width = window_width // 2
    assert half_width < median_dx
    if window_width % 2: # is odd
        window_slice = range(median_dx - half_width, median_dx + half_width + 1)
    else:
        window_slice = range(median_dx - half_width, median_dx + half_width)
    assert len(window_slice) == window_width
    return window_slice

def stride(stride_num, stride_length):
    return slice(stride_num*stride_length,(stride_num+1)*stride_length)

def awgn(arr, *, snr):
    """Add Gaussian white noise to arr."""
    return arr + np.sqrt(np.power(10.0, -snr/10)) * np.random.randn(*arr.shape)

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