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
    @staticmethod
    def fold_space(arr_timespace):
        n_space = arr_timespace.shape[1]
        is_odd = n_space % 2
        trunc_half = n_space // 2
        ix_first_half = range(trunc_half)
        ix_backwards_second_half = range(n_space,trunc_half+is_odd,-1)
        # TODO: remove expensive assertion
        assert np.array_equal(arr_timespace[:,ix_first_half],
                            arr_timespace[:,ix_backwards_second_half])
        return arr_timespace[:,:(trunc_half+is_odd)]

    def __init__(self, arr, sampling_rate=1):
        self._arr = self.fold_space(arr)
        # self.sampling_rate = sampling_rate
        # self.still_frames = np.apply_along_axis(lambda x: np.allclose(x,0),
        #     axis=1, self.time_differences)
        # self.plateaux = np.isclose(self.space_differences, 0)
        # self.space_relmax_ix = argrelmax(self._arr, axis=1)
        # self.space_relmin_ix = argrelmin(self._arr, axis=1)

    @cached_property
    def time_differences(self):
        return shift_subtract(self._arr, axis=0)
    @cached_property
    def space_differences(self):
        return shift_subtract(self._arr, axis=1)
    def clooge_bump_track(self):
        """
            This only works for the very simple case examined in Neuman 2015
        """
        # First cut out the initial rise of the bump.
        timespace = self._arr
        space_maxes = np.amax(timespace, axis=1)
        ix_time_with_max = np.argmax(space_maxes)
        timespace = timespace[ix_time_with_max:,:]
        # Then similarly cut the already calculated amplitudes
        amplitudes = space_maxes[ix_time_with_max:]
        # Find the change in amplitude through time
        delta_amplitudes = shift_subtract(peak_positions, axis=0)
        # When this amplitude stops changing, say the bump is dead
        # More properly we would say "close to zero," but there is a
        # certain resting activity.
        ix_time_bump_dies = np.where(np.isclose(delta_amplitudes),0)
        # Calculate the locations of the peak
        peak_positions = np.argmax(timespace, axis=1)
        # Calculate the velocities in terms of d index/d frame
        velocities = shift_subtract(peak_positions, axis=0)
