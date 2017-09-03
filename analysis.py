#!/usr/bin/env python3

"""
Author: Graham Smith

Versions:
    10 Jul 2017  Begin

Functions for the analysis of 1D data.
"""

import numpy as np

from scipy.signal import argrelmax, argrelmin
from cached_property import cached_property

import matplotlib.pyplot as plt

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

class TimeSpace(object):
    ''' Called TimeSpace because Time is the first dimension. '''
    @staticmethod
    def fold_space(arr_timespace):
        n_space = arr_timespace.shape[1]
        is_odd = n_space % 2
        trunc_half = n_space // 2
        ix_first_half = range(trunc_half)
        ix_backwards_second_half = range(n_space-1,trunc_half+is_odd-1,-1)
        # TODO: remove expensive assertion
        # assert np.allclose(arr_timespace[:,ix_first_half],
        #                     arr_timespace[:,ix_backwards_second_half])
        # IMPORTANT: the assertion was false. I don't know why.
        return arr_timespace[:,:(trunc_half+is_odd)]

    def __init__(self, arr, sampling_rate=1):
        self._arr = self.fold_space(arr)
        self.n_space = self._arr.shape[1]
        self.n_time = self._arr.shape[0]
        # self.sampling_rate = sampling_rate
        # self.still_frames = np.apply_along_axis(lambda x: np.allclose(x,0),
        #     axis=1, self.time_differences)
        # self.plateaux = np.isclose(self.space_differences, 0)
        # self.space_relmax_ix = argrelmax(self._arr, axis=1)
        # self.space_relmin_ix = argrelmin(self._arr, axis=1)

    # @cached_property
    # def time_differences(self):
    #     return shift_subtract(self._arr, axis=0)
    # @cached_property
    # def space_differences(self):
    #     return shift_subtract(self._arr, axis=1)
    def clooge_bump_properties(self):
        """
            This only works for the very simple case examined in Neuman 2015

            Looks at timeslice between timespace-peak and the half-peak
            disappearing.
        """
        # First cut out the initial rise of the bump.
        timespace = self._arr
        space_maxes = np.amax(timespace, axis=1)
        ix_time_with_max = np.argmax(space_maxes)
        timespace = timespace[ix_time_with_max:,:]
        n_time = timespace.shape[0]
        n_space = timespace.shape[1]
        # Then similarly cut the already calculated amplitudes
        amplitude = space_maxes[ix_time_with_max:]
        # Find the change in amplitude through time
        delta_amplitudes = shift_subtract(amplitude, axis=0)
        # When this amplitude stops changing, say the bump is dead
        # More properly we would say "when the amplitude is close to zero,"
        # but there is a certain resting activity.
        ix_time_bump_dies = np.nonzero(np.isclose(delta_amplitudes,
            0,atol=1e-5))[0]
        # Calculate the locations of the peak
        peak_ix = np.argmax(timespace, axis=1)
        # Calculate the velocity in terms of d index/d frame
        velocity = shift_subtract(peak_ix, axis=0)
        # Calculate the widths
        space_indices = np.tile(range(n_space), (n_time, 1))
        peak_ix_expanded = np.tile(peak_ix[...,np.newaxis],
                                            (1,n_space))
        assert any(map(any, np.logical_and(timespace <= (amplitude[...,np.newaxis] / 2),
                                space_indices < peak_ix_expanded)))
        # right_ix = nonzero_first(np.logical_and(
        #                         timespace <= (amplitude[...,np.newaxis] / 2),
        #                         space_indices > peak_ix_expanded),
        #                     axis=1)
        left_ix = nonzero_last(np.logical_and(
                                timespace <= (amplitude[...,np.newaxis] / 2),
                                space_indices < peak_ix_expanded),
                            axis=1)
        # Trim undefined widths
        if -1 in left_ix:
            first_undefined_width = np.nonzero(left_ix == -1)[0][0]
        else:
            first_undefined_width = len(left_ix)
        ix_defined_width = range(first_undefined_width)
        width = peak_ix[ix_defined_width] - left_ix[ix_defined_width]
        return {
            'width': width,
            'amplitude': amplitude[ix_defined_width],
            'peak_ix': peak_ix[ix_defined_width],
            'velocity': velocity[ix_defined_width[:-1]]
        }

def nonzero_last(arr, *, axis):
    """
        Takes N-d array (probably has to be 2-d?) and returns an array
        containing the indices of the last nonzero element of each
        slice along axis.
    """
    def nonzero_last_1d(arr):
        try:
            return np.nonzero(arr)[0][-1]
        except IndexError:
            return -1
    return np.apply_along_axis(nonzero_last_1d, axis, arr)

def nonzero_first(arr, *, axis):
    """
        Takes N-d array (probably has to be 2-d?) and returns an array
        containing the indices of the first nonzero element of each
        slice along axis.
    """
    def nonzero_first_1d(arr):
        try:
            return np.nonzero(arr)[0][0]
        except IndexError:
            return -1
    return np.apply_along_axis(nonzero_first_1d, axis, arr)