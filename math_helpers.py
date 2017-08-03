#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2 Aug 2017  Begin

import numpy as np

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