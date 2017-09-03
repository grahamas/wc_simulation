#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2 Aug 2017  Begin

import numpy as np

# def generator_solve_pre(prealloc, fn_generator, fn_diff, dt, n_time, y0, fn_reshape):
#     s = [slice(None)] * prealloc.ndim
#     s[0] = 0
#     print(s)
#     print(prealloc.shape)
#     prealloc[s] = fn_reshape(y0)
#     for y, time in fn_generator(fn_diff, dt, n_time, y0):
#         s[0] += 1
#         prealloc[s] = fn_reshape(y)
def generator_solve(fn_generator, fn_diff, dt, n_time, y0):
    output = np.zeros((n_time+1, len(y0)))
    output[0,:] = y0
    i_time = 0
    for y, time in fn_generator(fn_diff, dt, n_time, y0):
        output[i_time, :] = y
        i_time += 1
    return output

def euler_step(f, y, t, dt, *args):
    """
        Single step of Euler's method.
    """
    return y + dt * f(t, y, *args)

def euler(f, dt, n_time, y0, *args):
    """
        Integrates a differential equation, F, starting at time 0 and
        initial state y0 for n_time steps of increment dt.

        NOTE: The equation f must take the state of the previous time
        step and the current time.

        Uses Euler's method.
    """
    y = np.zeros((n_time+1, len(y0)))
    y[0,:] = y0
    for i_time in range(n_time):
        time = dt * i_time # TODO: Fix this.
        y[i_time+1] = euler_step(f, y[i_time,:], time, dt, *args)
    return y

def euler_generator(f, dt, n_time, y0, *args):
    """
        Integrates a differential equation, F, starting at time 0 and
        initial state y0 for n_time steps of increment dt.

        NOTE: The equation f must take the state of the previous time
        step and the current time.

        Uses Euler's method.

        Yields the result at each step (is a generator).
    """
    y = y0
    for i_time in range(n_time):
        t = dt * i_time # TODO: Fix this.
        y = euler_step(f, y, t, dt)
        yield(y, t)

def ode45_step(f, x, t, dt, *args):
    """
    One step of 4th Order Runge-Kutta method
    """
    k = dt
    k1 = k * f(t, x, *args)
    k2 = k * f(t + 0.5*k, x + 0.5*k1, *args)
    k3 = k * f(t + 0.5*k, x + 0.5*k2, *args)
    k4 = k * f(t + dt, x + k3, *args)
    return x + 1/6. * (k1 + k2 + k3 + k4)

def ode45(f, dt, n_time, y0, *args):
    """
    4th Order Runge-Kutta method
    """
    y = np.zeros((n_time+1, len(y0)))
    y[0,:] = y0
    for i_time in range(n_time):
        time = dt * i_time
        y[i_time+1,:] = ode45_step(f, y[i_time], time, dt, *args)
    return y

def ode45_generator(f, dt, n_time, y0, *args):
    """
    4th order Runge-Kutta method
    """
    y = y0
    for i_time in range(n_time):
        time = dt * i_time # TODO: Fix this.
        y = ode45_step(f, y, time, dt, *args)
        yield(y, time)

dct_generators = {
    'ode45': ode45_generator,
    'euler': euler_generator
}

dct_solvers = {
    'ode45': ode45,
    'euler': euler
}

# def ode45(f, t, x0, *args):
#     """
#     4th Order Runge-Kutta method
#     """
#     n = len(t)
#     x = np.zeros((n, len(x0)))
#     x[0] = x0
#     for i in range(n-1):
#         dt = t[i+1] - t[i]
#         x[i+1] = ode45_step(f, x[i], t[i], dt, *args)
#     return x
