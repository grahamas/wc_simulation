#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   7 Mar 2017  Begin
#   1 Jul 2017  Successfully replicated (Neuman 2015)

# Differences introduced from (Neuman 2015):
#   Edges of weight matrix are not halved
#       (see Neuman, p98, mid-page, a.k.a. lines 39-42 of JN's script)
#   Inhibitory sigmoid function is translated to S(0) = 0.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.verbose.set_level("helpful")
import json
import time
import os

import itertools
import functools

import sys
sys.path.append("..")
from plotting import LinesMovieFromSepPops, PlotDirectory

# json tools
def read_jsonfile(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params
def load_json_with(loader, pass_name=False):
    """
        A decorator for functions that take only a dict that can be
        loaded from a json. The decorator takes a function that loads
        the jsonfile as an argument.
    """
    def decorator(func=read_jsonfile):
        def wrapped(*args):
            if isinstance(args[0], str):
                loaded = loader(args[0])
                run_name = os.path.splitext(args[0])[0]
                return func(loaded, run_name) if pass_name else func(loaded)
            else:
                return func(*args)
        return wrapped
    return decorator
def get_neuman_params_from_json(json_filename):
    params = read_jsonfile(json_filename)
    run_name = os.path.splitext(json_filename)[0]
    n_space = params['space'][0]
    dx = params['space'][1]
    n_time = params['time'][0]
    dt = params['time'][1]
    input_duration = params['stimulus'][0]
    input_strength = params['stimulus'][1]
    input_width = params['stimulus'][2]
    e_dx = 0
    i_dx = 1
    sEE = params['s'][e_dx][e_dx]
    sEI = params['s'][e_dx][i_dx]
    sIE = params['s'][i_dx][e_dx]
    sII = params['s'][i_dx][i_dx]
    betaE = params['beta'][e_dx]
    betaI = params['beta'][i_dx]
    alphaE = params['alpha'][e_dx]
    alphaI = params['alpha'][i_dx]
    aE = params['a'][e_dx]
    aI = params['a'][i_dx]
    thetaE = params['theta'][e_dx]
    thetaI = params['theta'][i_dx]
    wEE = params['w'][e_dx][e_dx]
    wEI = params['w'][e_dx][i_dx]
    wIE = params['w'][i_dx][e_dx]
    wII = params['w'][i_dx][i_dx]
    tauE = params['tau'][e_dx]
    tauI = params['tau'][i_dx]
    noise_SNRE = params['noise_SNR'][e_dx]
    noise_SNRI = params['noise_SNR'][i_dx]
    mean_background_inputE = params['mean_background_input'][e_dx]
    mean_background_inputI = params['mean_background_input'][i_dx]
    params = {
            'space': [n_space, dx],
            'time': [n_time, dt],
            'stimulus': [input_duration, input_strength, input_width],
            's': [[sEE, sEI], [sIE, sII]],
            'beta': [betaE, betaI],
            'alpha': [alphaE, alphaI],
            'a': [aE, aI],
            'theta': [thetaE, thetaI],
            'w': [[wEE, wEI], [wIE, wII]],
            'tau': [tauE, tauI],
            'noise_SNR': [noise_SNRE, noise_SNRI],
            'mean_background_input': [mean_background_inputE,
                mean_background_inputI],
            'r': [1,1]
        }
    print('WARNING: Using Neuman equations without refraction.')
    return params


# From Jeremy's code. Not sure what paper it corresponds to.
def calculate_weight_matrix(dx, N, w, s):
    weight_mx = np.zeros((N, N))
    for i in range(N):
        weight_mx[i, :] = w*np.exp(-np.abs(dx*(np.arange(N)-i))/s)*dx/(2*s)
        # The division by 2*s normalizes the beta,
        # to separate the scaling w from the space constant s
    return weight_mx

def calculate_weight_matrices(dx, n_space, W, S):
    l_weights = []
    for dx1 in range(len(S)):
        row = []
        for dx2 in range(len(S[0])):
            row += [calculate_weight_matrix(dx, n_space, W[dx1][dx2],
                                                S[dx1][dx2])]
        l_weights += [row]
    return l_weights

def calculate_connectivity_mx(*args):
    # TODO: remove intermediate "calculate_weight_matrices"
    mx_list = calculate_weight_matrices(*args)
    return np.concatenate(
        [np.concatenate(row, axis=1) for row in mx_list],
        axis=0)

def awgn(arr, *, snr):
    return arr + np.sqrt(np.power(10.0, -snr/10)) * np.random.randn(*arr.shape)

def sigmoid(x, a, theta):
    return 1 / (1 + np.exp(-a * (x - theta)))

def sigmoid_norm(x, a, theta):
    return sigmoid(x,a,theta) - sigmoid(0,a,theta)

def sigmoid_norm_rectify(x,*, a, theta):
    return np.maximum(0, sigmoid_norm(x, a, theta))

def sigmoid_rectify(x,*, a, theta):
    return np.maximum(0, sigmoid(x,a,theta))

def central_window(total_len, window_width):
    median_dx = total_len // 2
    half_width = window_width // 2
    assert half_width < median_dx
    if window_width % 2: # is odd
        window_slice = range(median_dx - half_width, median_dx + half_width + 1)
    else:
        window_slice = range(median_dx - half_width, median_dx + half_width)
    assert len(window_slice) == window_width
    return window_slice

def makefn_interactive_simulate_neuman(json_filename):
    # The factory is necessary so that Jupyter's autoreload
    # functionality works correctly.
    with open(json_filename, 'r') as default_json:
        DEFAULT_PARAMS = json.load(default_json)
    # This is almost certainly the stupidest thing I've done while programming
    e_dx = 0
    i_dx = 1
    def interactive_simulate(
        run_name:str=os.path.splitext(json_filename)[0],
        n_space:int=DEFAULT_PARAMS['space'][0],
        dx:float=DEFAULT_PARAMS['space'][1],
        n_time:int=DEFAULT_PARAMS['time'][0],
        dt:float=DEFAULT_PARAMS['time'][1],
        input_duration:int=DEFAULT_PARAMS['stimulus'][0],
        input_strength:float=DEFAULT_PARAMS['stimulus'][1],
        input_width:int=DEFAULT_PARAMS['stimulus'][2],
        sEE:float=DEFAULT_PARAMS['s'][e_dx][e_dx],
        sEI:float=DEFAULT_PARAMS['s'][e_dx][i_dx],
        sIE:float=DEFAULT_PARAMS['s'][i_dx][e_dx],
        sII:float=DEFAULT_PARAMS['s'][i_dx][i_dx],
        betaE:float=DEFAULT_PARAMS['beta'][e_dx],
        betaI:float=DEFAULT_PARAMS['beta'][i_dx],
        alphaE:float=DEFAULT_PARAMS['alpha'][e_dx],
        alphaI:float=DEFAULT_PARAMS['alpha'][i_dx],
        aE:float=DEFAULT_PARAMS['a'][e_dx],
        aI:float=DEFAULT_PARAMS['a'][i_dx],
        thetaE:float=DEFAULT_PARAMS['theta'][e_dx],
        thetaI:float=DEFAULT_PARAMS['theta'][i_dx],
        wEE:float=DEFAULT_PARAMS['w'][e_dx][e_dx],
        wEI:float=DEFAULT_PARAMS['w'][e_dx][i_dx],
        wIE:float=DEFAULT_PARAMS['w'][i_dx][e_dx],
        wII:float=DEFAULT_PARAMS['w'][i_dx][i_dx],
        tauE:float=DEFAULT_PARAMS['tau'][e_dx],
        tauI:float=DEFAULT_PARAMS['tau'][i_dx],
        noise_SNRE:float=DEFAULT_PARAMS['noise_SNR'][e_dx],
        noise_SNRI:float=DEFAULT_PARAMS['noise_SNR'][i_dx],
        mean_background_inputE:float=DEFAULT_PARAMS['mean_background_input'][e_dx],
        mean_background_inputI:float=DEFAULT_PARAMS['mean_background_input'][i_dx]):
        params = {
            'space': [n_space, dx],
            'time': [n_time, dt],
            'stimulus': [input_duration, input_strength, input_width],
            's': [[sEE, sEI], [sIE, sII]],
            'beta': [betaE, betaI],
            'alpha': [alphaE, alphaI],
            'a': [aE, aI],
            'theta': [thetaE, thetaI],
            'w': [[wEE, wEI], [wIE, wII]],
            'tau': [tauE, tauI],
            'noise_SNR': [noise_SNRE, noise_SNRI],
            'mean_background_input': [mean_background_inputE,
                mean_background_inputI],
            'r': [1,1]
        }
        print('WARNING: Using Neuman equations without refraction.')
        make_simulation_movie(params, run_name)
    return interactive_simulate

# Use this in Jupyter notebook
def interactive_widget_neuman(widget_module, json_filename):
    if __debug__:
        print("Debugging")
    else:
        print("NOT debugging")
    fn_interactive_simulate = makefn_interactive_simulate_neuman(json_filename)
    return widget_module.interact_manual(fn_interactive_simulate,
                run_name=os.path.splitext(json_filename)[0],
                n_space=(101,10001,100),dx=(0.01,1,0.01),
                n_time=(241,1041,40), dt=(0.0001,0.015,0.001),
                input_duration=(15,200,1),input_strength=(0,10,0.01),input_width=(1,400,1),
                sEE=(2,3,0.1),sEI=(2,3,0.1),sIE=(2,3,0.1),sII=(2,3,0.1),
                betaE=(0.7,1.4,0.1), betaI=(0.7,1.4,0.1),
                tauE=(0.05,0.3,0.01), tauI=(0.05,1,0.01),
                alphaE=(.5,2,.1),alphaI=(.5,2,.1),
                aE=(.5,2,.1),aI=(.5,2,.1),
                thetaE=(2,3,.1),thetaI=(6,9,.1),
                wEE=(0,95,0.1),wEI=(-95,0,0.1),wIE=(0,95,0.1),wII=(-95,0,0.1),
                noise_SNRE=(75,115,1),noise_SNRI=(75,115,1),
                mean_background_inputE=(0,2,0.1), mean_background_inputI=(0,2,0.1))

def euler_generator(dt, n_time, y0, F):
    y = y0
    for i_time in range(n_time):
        y = y + dt*F(y, i_time)
        yield(y, i_time)

def makefn_wilson_cowan(vr_time_constant, vr_alpha, vr_beta,
    mx_connectivity, vr_current, fn_nonlinearity, fn_input):
    def wilson_cowan(activity, i_time):
        vr_stimulus = fn_input(i_time)
        return (-vr_alpha * activity + (1 - activity) * vr_beta\
            * fn_nonlinearity(mx_connectivity @ activity
                + vr_current + vr_stimulus))/vr_time_constant
    return wilson_cowan

def makefn_from_neuman(*, space, time, stimulus, s, beta, alpha, r,
    theta, a, w, tau, noise_SNR, mean_background_input):
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

    fn_sigmoid = functools.partial(sigmoid_norm_rectify,
                                        a=fn_expand_param_in_space(a),
                                        theta=fn_expand_param_in_space(theta))
    fn_noise = functools.partial(awgn, snr=fn_expand_param_in_space(noise_SNR))
    fn_nonlinearity = lambda x: fn_sigmoid(fn_noise(x))

    input_duration, input_strength, input_width = stimulus
    input_slice = central_window(n_space, input_width)
    blank_input = fn_expand_param_in_population(np.zeros(n_space))
    one_pop_stim =  np.zeros(n_space)
    one_pop_stim[input_slice] = one_pop_stim[input_slice] + input_strength
    vr_stimulus = fn_expand_param_in_population(one_pop_stim)

    def fn_input(i_time):
        if i_time <= input_duration:
            return vr_stimulus
        else:
            return blank_input

    return makefn_wilson_cowan(vr_time_constant, vr_alpha, vr_beta,
        mx_connectivity, vr_current, fn_nonlinearity, fn_input)

def simulate_neuman(*, space, time, **params):
    n_populations = 2
    n_space, dx = space
    n_time, dt = time

    activity = np.zeros((n_time, n_populations, n_space))
    y0 = np.concatenate((np.zeros(n_space), np.zeros(n_space)), axis=0)

    fn_wilson_cowan = makefn_from_neuman(space=space, time=time, **params)

    for y, i_time in euler_generator(dt, n_time, y0, fn_wilson_cowan):
        activity[i_time,:,:] = y.reshape((n_populations, n_space))

    return activity

@load_json_with(get_neuman_params_from_json, pass_name=True)
def make_simulation_movie(params, run_name):
    n_space, dx = params["space"]
    activity = simulate_neuman(**params)
    E = activity[:,0,:]
    I = activity[:,1,:]
    plt_dir = PlotDirectory(run_name, params)
    lE = LinesMovieFromSepPops([E, I, E+I],
            save_to=plt_dir.pathify('activity_movie.mp4'))
    plt.show()


