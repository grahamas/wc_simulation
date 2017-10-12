#!/usr/bin/env python3

"""
Author: Graham Smith

Versions:
    10 Jul 2017  Begin

Functions for the analysis of 1D data.
"""

import logging as log
log.basicConfig(level=log.DEBUG)

import numpy as np

from scipy.signal import argrelmax, argrelmin
#from cached_property import cached_property

import os
import json
import time
import pickle

# Enable local imports from parent directory
import sys
sys.path.append("..")

from plot import Plots
import math_aux as math

def one_pop_analysis(*, data, results_struct):
    rs = results_struct
    plots = rs.plots
    plots.clear()
    data = data[:,0,:]
    print('Plotting movie?')
    if len(plots.movie_params) > 0:
        print('Plotting movie.')
        plots.movie(movie_type='lines',
                lines_data=[data],
                save_to='activity_movie.mp4',
                xlabel='space (a.u.)', ylabel='amplitude (a.u.)',
                title='1D Wilson-Cowan Simulation')
    plots.imshow(data, xlabel='space (a.u.)', ylabel='time (a.u.)',
        title='Heatmap of activity', save_to='E_timespace.png')
    timespace_E = TimeSpace(data)
    bump_properties = timespace_E.clooge_bump_properties()
    peak_ix = bump_properties.pop('peak_ix')
    width = bump_properties.pop('width')
    amplitude = bump_properties.pop('amplitude')
    plots.multiline_plot([amplitude], ['amplitude'], xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Properties of moving bump',
        save_to='cloogy_bump_amp.png')
    plots.multiline_plot_from_dict(bump_properties, xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Properties of moving bump',
        save_to='cloogy_bump_vel.png')
    plots.multiline_plot([width], ['width'], xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Width of moving bump',
        save_to='cloogy_bump_width.png')
    rs.save_data(data=bump_properties, filename='bump_properties.pkl')

def e_i_analysis(*, data, results_struct):
    rs = results_struct
    plots = rs.plots
    E = data[:,0,:]
    I = data[:,1,:]
    plots.clear()
    log.info("Plotting EI movie?")
    if len(plots.movie_params) > 0:
        log.info("... yes.")
        plots.movie(movie_type="lines",
                lines_data=[E, I, E+I],
                save_to='activity_movie.mp4',
                xlabel='space (a.u.)', ylabel='amplitude (a.u.)',
                title='1D Wilson-Cowan Simulation')
    plots.imshow(E, xlabel='space (a.u.)', ylabel='time (a.u.)',
        title='Heatmap of E activity', save_to='E_timespace.png')
    timespace_E = TimeSpace(E)
    bump_properties = timespace_E.clooge_bump_properties()
    peak_ix = bump_properties.pop('peak_ix')
    width = bump_properties.pop('width')
    amplitude = bump_properties.pop('amplitude')
    plots.multiline_plot([amplitude], ['amplitude'], xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Properties of moving bump',
        save_to='cloogy_bump_amp.png')
    plots.multiline_plot_from_dict(bump_properties, xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Properties of moving bump',
        save_to='cloogy_bump_vel.png')
    plots.multiline_plot([width], ['width'], xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Width of moving bump',
        save_to='cloogy_bump_width.png')
    num_freeze_frames = 10 # TODO
    freeze_frame_step = E.shape[0] // num_freeze_frames
    plots.multiline_plot((E+I)[::freeze_frame_step,:], [], xlabel="space (a.u.)",
            ylabel = 'activity (a.u.)', title='Profile of activity at various time points',
            save_to='activity_profile.png')
    rs.save_data(data=bump_properties, filename='bump_properties.pkl')

class Results(object):
    # TODO: Bad, redo this.
    analysis_fn_dct = {
            'e_i': e_i_analysis,
            'one_pop': one_pop_analysis
            }

    def __init__(self, *, data, run_name, model_params, sep='_', root='plots',
            figure_params={}, analyses_dct={}):
        self.data = data
        self._run_name = run_name
        self._init_time = time.strftime("%Y%m%d{}%H%M%S".format(sep))
        self.model_params = model_params
        self.init_save(root, sep)
        self.analyses_dct = analyses_dct
        lattice = model_params['lattice']
        # TODO: rewrite plotting to use lattice  
        self.dx, self.space_max = lattice['space_step'], lattice['space_extent']
        self.dt, self.time_max = lattice['time_step'], lattice['time_extent']
        self.plots = Plots(results=self, **figure_params)

    def init_save(self, root, sep):
        dir_name = self._init_time + sep + self._run_name
        self._dir_path = os.path.join(root, dir_name)
        if os.path.exists(self._dir_path):
            raise Exception('"Uniquely" named folder already exists.')
        os.mkdir(self._dir_path)
        self.save_data(data=self.model_params, filename='params.json',
            mode='w', save_fn=lambda data, file: json.dump(data, file,
                sort_keys=True, indent=4))


    def save_data(self, *, data, filename, mode='wb', save_fn=pickle.dump):
        with open(self.pathify(filename), mode) as file:
            save_fn(data, file)

    def pathify(self, name, subdir=''):
        return os.path.join(self._dir_path, subdir, name)

    def analyse(self):
        log.info('Analysing...')
        for analysis_name, analysis_params in self.analyses_dct.items():
            self.analysis_fn_dct[analysis_name](data=self.data, 
                    results_struct=self, **analysis_params)        

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
        delta_amplitudes = math.shift_subtract(amplitude, axis=0)
        # When this amplitude stops changing, say the bump is dead
        # More properly we would say "when the amplitude is close to zero,"
        # but there is a certain resting activity.
        ix_time_bump_dies = np.nonzero(np.isclose(delta_amplitudes,
            0,atol=1e-5))[0]
        # Calculate the locations of the peak
        peak_ix = np.argmax(timespace, axis=1)
        # Calculate the velocity in terms of d index/d frame
        velocity = math.shift_subtract(peak_ix, axis=0)
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
