#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   7 Mar 2017  Begin
#

import numpy as np
import matplotlib.pyplot as plt
import json
import time


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


def simulate(params):
    #n_dims = 2
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
    #t = params['t']
    #sd = params['sd']
    w = params['w']
    tau = params['tau']

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
            B[input_duration-10:input_duration] += input_strength
        JE = w[e][e] * np.matmul(l_weights[e][e], E[:, i_time-1]) + B
        JI = -0.2 * w[i][i] * np.matmul(l_weights[i][i], E[:, i_time-1]) + B
        JE_noise = awgn(JE, 80)
        JI_noise = awgn(JI, 80)

        FE = sigmoid(JE_noise, a[e], theta[e])
        FI = sigmoid(JI_noise, a[i], theta[i])
        #FI = 1 / (1 + np.exp(-a[i] * (JInoise - theta[i])))

        FE = np.maximum(0, FE)
        FI = np.maximum(0, FI)

        E[:, i_time] = E[
            :, i_time-1] + dt * (-(alpha[e] * E[:, i-1]) + (1-E[:, i-1])*beta[e]*FE)/tau[e]
        I[:, i_time] = I[
            :, i_time-1] + dt * (-(alpha[i] * I[:, i-1]) + (1-I[:, i-1])*beta[i]*FI)/tau[i]

        max_dx = np.argmax(
            E[:, i_time] - I[:, i_time])
        max_wave_dx[0, i_time] = max_dx
        max_wave[0, i_time] = E[max_dx, i_time] - I[max_dx, i_time]
    for i_space in range(n_space):
        max_time_dx = np.argmax(
            E[i_space, :] -I[i_space, :])
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


def plot_code_p101(A, dx):
    l_time_point = [5, 40, 80, 120]
    N = A.shape[0]
    for i_time_point, time_point in enumerate(l_time_point):
        _plot_code_p101(A, dx, N, i_time_point+1, time_point)
    return


def read_params(fn):
    with open(fn, 'r') as f:
        params = json.load(f)
    return params

if __name__ == "__main__":
    param_file = 'params.json'
    params = read_params(param_file)
    E, I = simulate(params)
    _, dx = params['space']
    this_id = time.strftime("%Y%m%d-%H%M%S")
    plot_code_p101(E, dx)
    plots_dir = 'plots/'
    plt.savefig(plots_dir+'{}_E.png'.format(this_id))
    plt.close()
    plot_code_p101(I, dx)
    plt.savefig(plots_dir+'{}_I.png'.format(this_id))
