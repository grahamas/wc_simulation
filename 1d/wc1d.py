#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   7 Mar 2017  Begin
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.verbose.set_level("helpful")
import json
import time
import os

import sys
sys.path.append("..")
from plotting import LinesMovie, PlotDirectory

e_dx = 0
i_dx = 1

# From Jeremy's code. Not sure what paper it corresponds to.
def calculate_weight_matrix(dx, N, w, s):
    weight_mx = np.zeros((N, N))
    for i in range(N):
        # Note divided by 2*s, which is not in the original 1973 paper
        # This normalizes the beta, to separate the scaling w from the
        # space constant s
        weight_mx[i, :] = w*np.exp(-np.abs(dx*(np.arange(N)-i))/s)*dx/(2*s)
    # This was in Jeremy's code, but I don't know why
    #weight_mx[0, :] = weight_mx[0, :] / 2
    #weight_mx[-1, :] = weight_mx[-1, :] / 2
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

def awgn(arr, snr):
    return arr + np.sqrt(np.power(10.0, -snr/10)) * np.random.randn(*arr.shape)

def sigmoid(x, a, theta):
    return 1 / (1 + np.exp(-a * (x - theta)))

def sigmoid_norm(x, a, theta):
    return sigmoid(x,a,theta) - sigmoid(0,a,theta)

def sigmoid_norm_rectify(x, a, theta):
    return np.maximum(0, sigmoid_norm(x, a, theta))

def sigmoid_rectify(x, a, theta):
    return np.maximum(0, sigmoid(x,a,theta))

def fnmake_interactive_simulate(json_filename):
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
        input_duration:int=DEFAULT_PARAMS['input'][0],
        input_strength:float=DEFAULT_PARAMS['input'][1],
        input_width:int=DEFAULT_PARAMS['input'][2],
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
        arg_dict = {
            'space': [n_space, dx],
            'time': [n_time, dt],
            'input': [input_duration, input_strength, input_width],
            's': [[sEE, sEI], [sIE, sII]],
            'beta': [betaE, betaI],
            'alpha': [alphaE, alphaI],
            'a': [aE, aI],
            'theta': [thetaE, thetaI],
            'w': [[wEE, wEI], [wIE, wII]],
            'tau': [tauE, tauI],
            'noise_SNR': [noise_SNRE, noise_SNRI],
            'mean_background_input': [mean_background_inputE,
                mean_background_inputI]
        }
        E, I = simulate(arg_dict)
        plt_dir = PlotDirectory(run_name, arg_dict)
        lE = LinesMovie([E, I, E+I], save_to=plt_dir.pathify('_E_movie.mp4'))
        plt.show()
    return interactive_simulate

# Use this in Jupyter notebook
def interactive_widget(widget_module, json_filename):
    if __debug__:
        print("Debugging")
    else:
        print("NOT debugging")
    fn_interactive_simulate = fnmake_interactive_simulate(json_filename)
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
                wEE=(0,95,0.1),wEI=(0,95,0.1),wIE=(0,95,0.1),wII=(0,95,0.1),
                noise_SNRE=(75,115,1),noise_SNRI=(75,115,1),
                mean_background_inputE=(0,2,0.1), mean_background_inputI=(0,2,0.1))

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

def simulate_from_json(json_filename):
    with open(json_filename, 'r') as default_json:
        params = json.load(default_json)
    run_name = os.path.splitext(json_filename)[0]
    n_space = params['space'][0]
    dx = params['space'][1]
    n_time = params['time'][0]
    dt = params['time'][1]
    input_duration = params['input'][0]
    input_strength = params['input'][1]
    input_width = params['input'][2]
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
    arg_dict = {
            'space': [n_space, dx],
            'time': [n_time, dt],
            'input': [input_duration, input_strength, input_width],
            's': [[sEE, sEI], [sIE, sII]],
            'beta': [betaE, betaI],
            'alpha': [alphaE, alphaI],
            'a': [aE, aI],
            'theta': [thetaE, thetaI],
            'w': [[wEE, wEI], [wIE, wII]],
            'tau': [tauE, tauI],
            'noise_SNR': [noise_SNRE, noise_SNRI],
            'mean_background_input': [mean_background_inputE,
                mean_background_inputI]
        }
    E, I = simulate(arg_dict)
    plt_dir = PlotDirectory(run_name, arg_dict)
    lE = LinesMovie([E, I, E+I], save_to=plt_dir.pathify('_E_movie.mp4'))
    plt.show()
def simulate(params):
    # n_dims = 2

    n_space, dx = params['space']
    n_time, dt = params['time']
    input_duration, input_strength, input_width = params['input']
    s = params['s']
    beta = params['beta']
    alpha = params['alpha']
    a = params['a']
    theta = params['theta']
    # t = params['t']
    # sd = params['sd']
    w = params['w']
    tau = params['tau']
    noise_SNR = params['noise_SNR']
    mean_background_input = params['mean_background_input']

    E = np.zeros((n_space, n_time))
    I = np.zeros((n_space, n_time))

    max_wave = np.zeros((1, n_time))
    max_wave_dx = np.zeros((1, n_time))
    max_wave_in_time = np.zeros((n_space, 1))
    max_wave_in_time_dx = np.zeros((n_space, 1))

    l_weights = calculate_weight_matrices(dx, n_space, w, s)

    input_slice = central_window(n_space, input_width)

    for i_time in range(2, n_time):
        #import pdb; pdb.set_trace()
        external_input = np.zeros((n_space))
        if i_time < input_duration:
            external_input[input_slice] += input_strength
        # TODO: Get rid of E-I system.
        # NOTE: As written, subscripts reverse of Neumann code
        JE = l_weights[e_dx][e_dx] @ E[:, i_time-1]\
            - l_weights[e_dx][i_dx] @ I[:, i_time-1]\
            + external_input + mean_background_input[e_dx]

        JI = -l_weights[i_dx][i_dx] @ I[:, i_time-1]\
             + l_weights[i_dx][e_dx] @ E[:, i_time-1]\
             + external_input + mean_background_input[i_dx]
        JE_noise = awgn(JE, noise_SNR[e_dx])
        JI_noise = awgn(JI, noise_SNR[i_dx])

        # TODO: Introduce scaling parameter
        FE = sigmoid_norm_rectify(JE_noise, a[e_dx], theta[e_dx])
        FI = sigmoid_rectify(JI_noise, a[i_dx], theta[i_dx])

        E[:, i_time] = E[:, i_time-1]\
                + dt * (-(alpha[e_dx] * E[:, i_time-1]) \
                + (1-E[:, i_time-1])*beta[e_dx]*FE)/tau[e_dx]
        I[:, i_time] = I[:, i_time-1] \
                + dt * (-(alpha[i_dx] * I[:, i_time-1]) \
                + (1-I[:, i_time-1])*beta[i_dx]*FI)/tau[i_dx]

        max_dx = np.argmax(E[:, i_time] - I[:, i_time])
        max_wave_dx[0, i_time] = max_dx
        max_wave[0, i_time] = E[max_dx, i_time] - I[max_dx, i_time]
    for i_space in range(n_space):
        max_time_dx = np.argmax(
            E[i_space, :] - I[i_space, :])
        max_wave_in_time[i_space, 0] = E[
            i_space, max_time_dx] - I[i_space, max_time_dx]
        max_wave_in_time_dx[i_space, 0] = max_time_dx

    return E, I

def _plot_code_p101(A, dx, N, subplot, time_point):
    plt.subplot(2, 2, subplot)
    plt.title(time_point)
    plt.plot(dx * np.arange(N), A[:, time_point])
    plt.ylim(-0.1, 0.5)
    plt.xlim(0, 100)

def plot_code_p101(A, dx, l_time_point=[5, 40, 80, 120]):
    N = A.shape[0]
    for i_time_point, time_point in enumerate(l_time_point):
        _plot_code_p101(A, dx, N, i_time_point+1, time_point)
    return

def read_params(fn):
    with open(fn, 'r') as f:
        params = json.load(f)
    return params

if __name__ == "__main__":
    run_name = 'FromJack'
    save_suffix = ''
    param_file = 'params_{}.json'.format(run_name)
    params = read_params(param_file)
    E, I = simulate(params)
    _, dx = params['space']
    l_time_point = [5, 70, 180, 280]
    plt_dir = PlotDirectory(run_name + save_suffix, params)
    plot_code_p101(E, dx, l_time_point)
    plot_prefix = 'NeumannFig2pt22_Custom_Points'
    plt_dir.savefig(plot_prefix+'_E.png')
    plt.close()
    plot_code_p101(I, dx, l_time_point)
    plt_dir.savefig(plot_prefix + '_I.png')
    step = 3
    E_movie = LinesMovie([E[::step], I[::step], E[::step]+I[::step]], save_to=plt_dir.pathify(plot_prefix+'_E_movie.mp4'), lines=True)
