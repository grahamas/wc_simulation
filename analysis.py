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

def shift_op(arr, *, op, axis):
    n_dims = len(arr.shape)
    left = ([slice(None)] * n_dims)
    left[axis] = range(len(arr)-1)
    right = ([slice(None)] * n_dims)
    right[axis] = range(1,len(arr))
    return op(arr[left], arr[right])

def shift_subtract(arr, *, axis):
    return shift_op(arr, op=np.subtract, axis=axis)

def shift_multiply(arr, *, axis):
    return shift_op(arr, op=np.multiply, axis=axis)

class Space(object):
    def __init__(self, frame):
        self.frame = frame
    def calc_bumps(self):
        arr_peak_ix = argrelmax(self.frame)
        local_minima_ix = argrelmin(self.frame)
        def bump_args(peak_ix):
            height = self.frame[peak_ix]
            half_height = height / 2
            lower_ix = np.where(self.frame < half_height)
            left_ix = np.max(lower_ix[lower_ix < peak_ix]) + 1
            right_ix = np.min(lower_ix[lower_ix > peak_ix]) - 1
            if np.any(np.logical_and(local_minima_ix >= left_ix,
                local_minima_ix < peak_ix)) or np.any(np.logical_and(
                    local_minima_ix <= right_ix,
                    local_minima_ix > peak_ix)):
                half_width = None
            else:
                half_width = right_ix - left_ix
            return {
                'peak_ix'   : peak_ix,
                'half_width': half_width,
                'height'    : height
            }
        self.bumps = [BumpFrame(**args) for args in map(bump_args, arr_peak_ix)]

class BumpFrame(object):
    def __init__(self, *, peak_ix=None,
            half_width=None, height=None):
        self.peak_ix = peak_ix
        assert half_width is not None and half_width >= 0
        self.half_width = half_width
        self.height = height

class Bump(object):
    def __init__(self, *, first_frame, first_frame_ix):
        self.frames = [first_frame]
        self.first_frame_ix = first_frame_ix
        self.velocity = []
    def add_frame(self, new_frame):
        self.frames += [new_frame]
        self.velocity += [new_frame.peak_ix - self.frames[-2].peak_ix]

class SpaceTime(object):
    def __init__(self, arr, sampling_rate=1):
        self._arr = arr
        self.sampling_rate = sampling_rate
        self.still_frames = np.apply_along_axis(lambda x: np.allclose(x,0),
            axis=1, self.time_differences)
        self.plateaux = np.isclose(self.space_differences, 0)
        self.space_relmax_ix = argrelmax(self._arr, axis=1)
        self.space_relmin_ix = argrelmin(self._arr, axis=1)

    @cached_property
    def time_differences(self):
        return shift_subtract(self._arr, axis=0)
    @cached_property
    def space_differences(self):
        return shift_subtract(self._arr, axis=1)
