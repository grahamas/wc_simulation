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

from plotting import LineMovie, PlotDirectory

def calculate_weight_matrix(dx, N, s):
    weight_mx = np.zeros((N, N))
    for i in range(N):
        weight_mx[:, i] = np.exp(-np.abs(dx*(np.arange(N)-i))/s)*dx/(2*s)
    # TODO: Why?
    weight_mx[:, [0, -1]] = weight_mx[:, [0, -1]] / 2
    return weight_mx


def calculate_weight_matrices(dx, n_space, S):
    l_weights = []
    for dx1 in range(len(S)):
        row = []
        for dx2 in range(len(S[0])):
            row += [calculate_weight_matrix(dx, n_space, S[dx1][dx2])]
        l_weights += [row]
    return l_weights


def awgn(arr, snr):
    return arr + np.sqrt(np.power(10.0, -snr/10)) * np.random.randn(*arr.shape)


def sigmoid(x, a, theta):
    return 1 / (1 + np.exp(-a * (x - theta))
                ) - 1 / (1+np.exp(a * theta))


def sigmoid_rectify(x, a, theta):
    return np.maximum(0, sigmoid(x, a, theta))

def interactive_simulate(run_name:str="Interactive",
    n_space:int=201, dx:float=0.5, n_time:int=381, dt:float=0.01,
    input_duration:int=55, input_strength:float=9.6,
    sEE:float=2.5, sEI:float=2.7, sIE:float=2.7, sII:float=2.5, 
    betaE:float=1.1, betaI:float=1.1,
    alphaE:float=1.2, alphaI:float=1.0, 
    aE:float=1.2, aI:float=1.0,
    thetaE:float=2.6, thetaI:float=8.0,
    wEE:float=16.0, wEI:float=91.0, wIE:float=27.0, wII:float=20.0,
    tauE:float=0.1, tauI:float=0.18,
    noise_SNRE:float=100, noise_SNRI:float=100):
    arg_dict = {
        'space': [n_space, dx],
        'time': [n_time, dt],
        'input': [input_duration, input_strength],
        's': [[sEE, sEI], [sIE, sII]],
        'beta': [betaE, betaI],
        'alpha': [alphaE, alphaI],
        'a': [aE, aI],
        'theta': [thetaE, thetaI],
        'w': [[wEE, wEI], [wIE, wII]],
        'tau': [tauE, tauI],
        'noise_SNR': [noise_SNRE, noise_SNRI]
    }
    E, I = simulate(arg_dict)
    plt_dir = PlotDirectory(run_name, arg_dict)
    lE = LineMovie(E, save_to=plt_dir.pathify('_E_movie.mp4'))
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
    e = 0
    i = 1

    n_space, dx = params['space']
    n_time, dt = params['time']
    input_duration, input_strength = params['input']
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

    E = np.zeros((n_space, n_time))
    I = np.zeros((n_space, n_time))

    max_wave = np.zeros((1, n_time))
    max_wave_dx = np.zeros((1, n_time))
    max_wave_in_time = np.zeros((n_space, 1))
    max_wave_in_time_dx = np.zeros((n_space, 1))

    l_weights = calculate_weight_matrices(dx, n_space, s)

    for i_time in range(1, n_time):
        B = np.zeros((n_space))
        if i_time < input_duration:
            B[2*48:2*52] += input_strength
        # TODO: Get rid of E-I system.
        # NOTE: As written, subscripts reverse of Neumann code
        JE = w[e][e] * np.matmul(l_weights[e][e], E[:, i_time-1]) + B + 0.1 -\
             w[e][i] * np.matmul(l_weights[e][i], I[:, i_time-1])
        # 0.1
        # TODO: Get rid of arbitrary constants
        JI = -0.2 * w[i][i] * np.matmul(l_weights[i][i], I[:, i_time-1]) + 0.1 + \
            B +\
             w[i][e] * np.matmul(l_weights[i][e], E[:, i_time-1])
             # 0.1\ 
        JE_noise = awgn(JE, noise_SNR[e])
        JI_noise = awgn(JI, noise_SNR[i])

        FE = sigmoid_rectify(JE_noise, a[e], theta[e])
        FI = sigmoid_rectify(JI_noise, a[i], theta[i])
        # FI = 1 / (1 + np.exp(-a[i] * (JI_noise - theta[i])))

        E[:, i_time] = E[
            :, i_time-1] + dt * (-(alpha[e] * E[:, i-1]) +
                                 (1-E[:, i-1])*beta[e]*FE)/tau[e]
        I[:, i_time] = I[
            :, i_time-1] + dt * (-(alpha[i] * I[:, i-1]) +
                                 (1-I[:, i-1])*beta[i]*FI)/tau[i]

        max_dx = np.argmax(
            E[:, i_time] - I[:, i_time])
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
    E_movie = LineMovie(E, save_to=plt_dir.pathify(plot_prefix+'_E_movie.mp4'))
