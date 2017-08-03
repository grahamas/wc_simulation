#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   7 Mar 2017  Begin
#   1 Jul 2017  Successfully replicated Neuman 2015
#   6 Jul 2017  Abstracted WC implementation to passing function to solver
#   6 Jul 2017  Made entire calculation matrix ops, no explicit populations

# Differences introduced from (Neuman 2015): NO LONGER TRUE?
#   Edges of weight matrix are not halved
#       (see Neuman, p98, mid-page, a.k.a. lines 39-42 of JN's script)
#   Inhibitory sigmoid function is translated to S(0) = 0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.verbose.set_level("helpful")
import json
import time
import os
import scipy

import itertools
import functools

import sys
sys.path.append("..")
from plotting import LinesMovieFromSepPops, ResultInfo, ResultPlots
from analysis import TimeSpace
import math_helpers as mh

#region json helpers
def read_jsonfile(filename):
    """Reads jsonfile of name filename and returns contents as dict."""
    with open(filename, 'r') as f:
        params = json.load(f)
    return params
def load_json_with(loader, pass_name=False):
    """
        A decorator for functions whose first argument is a dict
        that could be loaded from a json-file.

        If the first argument is a string rather than a dict, the wrapper
        tries to load a file with that string name and then pass the
        loaded dict (along with the filename, optionally) to the
        wrapped function.
    """
    def decorator(func=read_jsonfile):
        def wrapper(*args, **kwargs):
            if isinstance(args[0], str):
                loaded = loader(args[0])
                if pass_name:
                    run_name = os.path.splitext(args[0])[0]
                    out_args = [loaded, run_name, *args[1:]]
                else:
                    out_args = [loaded, *args[1:]]
                return func(*out_args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
def get_neuman_params_from_json(json_filename):
    params = read_jsonfile(json_filename)
    run_name = os.path.splitext(json_filename)[0]
    params['r'] = [1,1]
    print('WARNING: Using Neuman equations without refraction.')
    return params
#endregion

#region calculate weight matrices
def calculate_weight_matrix(dx, N, w, s):
    """
        Calculates a weight matrix for the case of 1D WC with
        exponentially decaying spatial connectivity.
    """
    weight_mx = np.zeros((N, N))
    for i in range(N):
        weight_mx[i, :] = w*np.exp(-np.abs(dx*(np.arange(N)-i))/s)*dx/(2*s)
        # The division by 2*s normalizes the beta,
        # to separate the scaling w from the space constant s
    return weight_mx
def calculate_weight_matrices(dx, n_space, W, S):
    """
        Given N x N matrix of population interaction weights (and sigmas),
        this calculates the convolution matrix for each, returning
        an NxN matrix of n_space x n_space convolution matrices.
    """
    l_weights = []
    for dx1 in range(len(S)):
        row = []
        for dx2 in range(len(S[0])):
            row += [calculate_weight_matrix(dx, n_space, W[dx1][dx2],
                                                S[dx1][dx2])]
        l_weights += [row]
    return l_weights
def calculate_connectivity_mx(*args):
    """
        This concatenates the list of matrices returned by
        calculate_weight_matrices.
    """
    # TODO: remove intermediate "calculate_weight_matrices"
    mx_list = calculate_weight_matrices(*args)
    return np.concatenate(
        [np.concatenate(row, axis=1) for row in mx_list],
        axis=0)
#endregion

def makefn_neuman_implementation(*, space, time, stimulus, nonlinearity, s,
    beta, alpha, r, w, tau, noise_SNR, mean_background_input, noiseless=False):
    """
        Returns a function that implements the Wilson-Cowan equations
        as parametrized in Neuman 2015.

        The RETURNED function takes the previous activity state and the
        current time index as arguments, returning the dy/dt at the
        current time index.
    """
    n_space, dx = space
    n_time, dt = time
    n_population = len(tau)

    fn_expand_param_in_space = lambda x: np.repeat(x, n_space)
    fn_expand_param_in_population = lambda x: np.tile(x, n_population)

    vr_time_constant = fn_expand_param_in_space(tau)
    #vr_decay = np.array(alpha)
    #vr_refraction = np.array(r)
    vr_alpha = fn_expand_param_in_space(alpha)
    vr_beta = fn_expand_param_in_space(beta)
    mx_connectivity = calculate_connectivity_mx(dx, n_space, w, s)
    vr_current = fn_expand_param_in_space(mean_background_input)

    fn_nonlinearity = mh.NONLINEARITIES[nonlinearity['name']]
    dct_nl_args = {k: fn_expand_param_in_space(v)
        for k, v in nonlinearity['args'].items()}
    fn_transfer = functools.partial(fn_nonlinearity, **dct_nl_args)
    fn_noise = functools.partial(mh.awgn,
        snr=fn_expand_param_in_space(noise_SNR))
    if not noiseless:
        fn_nonlinearity = lambda x: fn_sigmoid(fn_noise(x))
    else:
        fn_nonlinearity = lambda x: fn_sigmoid(x)

    input_duration, input_strength, input_width = stimulus
    input_slice = mh.central_window(n_space, input_width)
    blank_input = fn_expand_param_in_population(np.zeros(n_space))
    one_pop_stim =  np.zeros(n_space)
    one_pop_stim[input_slice] = one_pop_stim[input_slice] + input_strength
    stimulus_input = fn_expand_param_in_population(one_pop_stim)

    def fn_input(t):
        if t <= input_duration*dt:
            return stimulus_input
        else:
            return blank_input

    def neuman_implementation(t, activity):
        vr_stimulus = fn_input(t)
        return (-vr_alpha * activity + (1 - activity) * vr_beta\
            * fn_nonlinearity(mx_connectivity @ activity
                + vr_current + vr_stimulus)) / vr_time_constant

    return neuman_implementation
def simulate_neuman(*, space, time, **params):
    """
        Simulates the Wilson-Cowan equation using Neuman's 2015
        parametrization and Euler's method (also as in Neuman 2015).
    """
    n_populations = 2
    max_space, dx = space
    max_time, dt = time
    n_space = int(max_space / dx)
    n_time = int(max_time / dt)
    space = [n_space, dx]
    time = [n_time, dt]

    input_duration, input_strength, input_width = params['stimulus']
    input_duration = int(input_duration // dt)
    input_width = int(input_width // dx)
    params['stimulus'] = [input_duration, input_strength, input_width]

    activity = np.zeros((n_time, n_populations, n_space))
    y0 = np.concatenate((np.zeros(n_space), np.zeros(n_space)), axis=0)

    solver_name = params.pop('solver')
    generator = SOLVERS[solver_name]

    fn_wilson_cowan = makefn_neuman_implementation(space=space, time=time,
        **params)

    print('starting simulation...')
    i_time = 0
    for y, time in generator(fn_wilson_cowan, dt, n_time, y0):
        activity[i_time,:,:] = y.reshape((n_populations, n_space))
        i_time += 1
    print('simulation done.')

    return activity

@load_json_with(get_neuman_params_from_json, pass_name=True)
def run_simulation(params, run_name, modifications=None, show_figs=False,
    movie_show=None, timespace_show=None, skip_movies=False):
    """
        Run the simulation resulting from simulate_neuman, and
        save results.
    """
    if movie_show is None:
        movie_show = show_figs
    if timespace_show is None:
        timespace_show = show_figs
    if modifications:
        for key, value in modifications.items():
            params[key] = value

    activity = simulate_neuman(**params)
    E = activity[:,0,:]
    I = activity[:,1,:]
    result_info = ResultInfo(run_name, params)
    result_plots = ResultPlots(result_info, show=show_figs)
    plt.clf()
    if not skip_movies:
        result_plots.movie(movie_class=LinesMovieFromSepPops,
                lines_data=[E, I, E+I],
                save_to='activity_movie.mp4',
                xlabel='space (a.u.)', ylabel='amplitude (a.u.)',
                title='1D Wilson-Cowan Simulation', clear=(not movie_show))
        result_plots.movie(movie_class=LinesMovieFromSepPops,
                data=[np.transpose(E), np.transpose(I), np.transpose(E+I)],
                save_to='activity_space_frames.mp4',
                xlabel='time (a.u.)', ylabel='amplitude (a.u.)',
                title='1D Wilson-Cowan Simulation', clear=(not movie_show))
    result_plots.imshow(E, xlabel='space (a.u.)', ylabel='time (a.u.)',
        title='Heatmap of E activity', save_to='E_timespace.png',
        clear=(not timespace_show))
    timespace_E = TimeSpace(E)
    bump_properties = timespace_E.clooge_bump_properties()
    peak_ix = bump_properties.pop('peak_ix')
    width = bump_properties.pop('width')
    amplitude = bump_properties.pop('amplitude')
    result_plots.multiline_plot([amplitude], ['amplitude'], xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Properties of moving bump',
        save_to='cloogy_bump_amp.png')
    result_plots.multiline_plot_from_dict(bump_properties, xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Properties of moving bump',
        save_to='cloogy_bump_vel.png')
    result_plots.multiline_plot([width], ['width'], xlabel='time (a.u.)',
        ylabel='(a.u., various)', title='Width of moving bump',
        save_to='cloogy_bump_width.png')
    result_info.save_data(data=bump_properties, filename='bump_properties.pkl')
