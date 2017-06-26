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

DEFAULT_JSON_FILENAME = 'params_FromJack.json'

# From Jeremy's code. Not sure what paper it corresponds to.
def calculate_weight_matrix(dx, N, w, s):
    weight_mx = np.zeros((N, N))
    for i in range(N):
        # Note divided by 2*s, which is not in the original 1973 paper
        # This normalizes the beta, to separate the scaling w from the space constant s
        weight_mx[:, i] = w*np.exp(-np.abs(dx*(np.arange(N)-i+1))/s)*dx/(2*s)
    # TODO: Why?
    # This was in Jeremy's code, but I can't justify keeping it
    # weight_mx[:, [0, -1]] = weight_mx[:, [0, -1]] / 2
    return weight_mx


def calculate_weight_matrices(dx, n_space, W, S):
    l_weights = []
    for dx1 in range(len(S)):
        row = []
        for dx2 in range(len(S[0])):
            row += [calculate_weight_matrix(dx, n_space, W[dx1][dx2], S[dx1][dx2])]
        l_weights += [row]
    return l_weights


def awgn(arr, snr):
    return arr + np.sqrt(np.power(10.0, -snr/10)) * np.random.randn(*arr.shape)


def sigmoid(x, a, theta):
    return 1 / (1 + np.exp(-a * (x - theta))
                ) - 1 / (1+np.exp(a * theta))


def sigmoid_rectify(x, a, theta):
    return np.maximum(0, sigmoid(x, a, theta))

# This is almost certainly the stupidest thing I've done while programming
e = 0
i = 1
with open(DEFAULT_JSON_FILENAME, 'r') as default_json: 
    DEFAULT_PARAMS = json.load(default_json)
def interactive_simulate(run_name:str="Interactive",
    n_space:int=DEFAULT_PARAMS['space'][0], 
    dx:float=DEFAULT_PARAMS['space'][1], 
    n_time:int=DEFAULT_PARAMS['time'][0], 
    dt:float=DEFAULT_PARAMS['time'][1],
    input_duration:int=DEFAULT_PARAMS['input'][0], 
    input_strength:float=DEFAULT_PARAMS['input'][1],
    input_width:int=DEFAULT_PARAMS['input'][2],
    sEE:float=DEFAULT_PARAMS['s'][e][e], 
    sEI:float=DEFAULT_PARAMS['s'][e][i], 
    sIE:float=DEFAULT_PARAMS['s'][i][e], 
    sII:float=DEFAULT_PARAMS['s'][i][i], 
    betaE:float=DEFAULT_PARAMS['beta'][e], 
    betaI:float=DEFAULT_PARAMS['beta'][i],
    alphaE:float=DEFAULT_PARAMS['alpha'][e], 
    alphaI:float=DEFAULT_PARAMS['alpha'][i], 
    aE:float=DEFAULT_PARAMS['a'][e], 
    aI:float=DEFAULT_PARAMS['a'][i],
    thetaE:float=DEFAULT_PARAMS['theta'][e], 
    thetaI:float=DEFAULT_PARAMS['theta'][i],
    wEE:float=DEFAULT_PARAMS['w'][e][e], 
    wEI:float=DEFAULT_PARAMS['w'][e][i], 
    wIE:float=DEFAULT_PARAMS['w'][i][e], 
    wII:float=DEFAULT_PARAMS['w'][i][i],
    tauE:float=DEFAULT_PARAMS['tau'][e], 
    tauI:float=DEFAULT_PARAMS['tau'][i],
    noise_SNRE:float=DEFAULT_PARAMS['noise_SNR'][e], 
    noise_SNRI:float=DEFAULT_PARAMS['noise_SNR'][i],
    mean_background_inputE:float=DEFAULT_PARAMS['mean_background_input'][e], 
    mean_background_inputI:float=DEFAULT_PARAMS['mean_background_input'][i]):
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

# interact_manual(interactive_simulate, run_name='interactive',
#                 n_space=(101,1001,100),
#                 dx=(0.01,1,0.01), n_time=(241,1041,40), dt=(0.0001,0.1,0.01),
#                 input_duration=(20,200,5),input_strength=(5,15,0.1),
#                 sEE=(2,3,0.1),sEI=(2,3,0.1),sIE=(2,3,0.1),sII=(2,3,0.1),
#                 betaE=(0.7,1.4,0.1), betaI=(0.7,1.4,0.1),
#                 tauE=(0.05,0.3,0.01), tauI=(0.05,0.3,0.01),
#                 alphaE=(.5,2,.1),alphaI=(.5,2,.1),
#                 aE=(.5,2,.1),aI=(.5,2,.1),
#                 thetaE=(2,3,.1),thetaI=(6,9,.1),
#                 wEE=(10,20,1),wEI=(20,30,1),wIE=(85,95,1),wII=(15,25,1),
#                 noise_SNRE=(75,85,1),noise_SNRI=(75,85,1))

def simulate(params):
    # n_dims = 2
    e_dx = 0
    i_dx = 1

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

    median_dx = n_space // 2
    half_input = input_width // 2
    assert half_input < median_dx
    if input_width % 2:
        input_slice = range(median_dx - half_input, median_dx + half_input)
    else:
        input_slice = range(median_dx - half_input, median_dx + half_input - 1)

    for i_time in range(2, n_time):
        external_input = np.zeros((n_space))
        if i_time < input_duration:
            external_input[input_slice] += input_strength
        # TODO: Get rid of E-I system.
        # NOTE: As written, subscripts reverse of Neumann code
        JE = np.matmul(l_weights[e_dx][e_dx], E[:, i_time-1])\
            - np.matmul(l_weights[e_dx][i_dx], I[:, i_time-1])\
            + external_input + mean_background_input[e_dx]

        JI = -np.matmul(l_weights[i_dx][i_dx], I[:, i_time-1])\
             + np.matmul(l_weights[i_dx][e_dx], E[:, i_time-1])\
             + external_input + mean_background_input[i_dx]
        JE_noise = awgn(JE, noise_SNR[e_dx])
        JI_noise = awgn(JI, noise_SNR[i_dx])

        # TODO: Introduce scaling parameter
        FE = sigmoid_rectify(JE_noise, a[e_dx], theta[e_dx])
        FI = sigmoid_rectify(JI_noise, a[i_dx], theta[i_dx])
        #FI = np.maximum(0, 1 / (1 + np.exp(-a[i] * (JI_noise - theta[i]))))

        E[:, i_time] = E[:, i_time-1]\
                + dt * (-(alpha[e_dx] * E[:, i_time-1]) \
                + (1-E[:, i_time-1])*FE)/tau[e]
        I[:, i_time] = I[:, i_time-1] \
                + dt * (-(alpha[i_dx] * I[:, i_time-1]) \
                + (1-I[:, i_time-1])*FI)/tau[i]

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
