#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2 Aug 2017  Begin
#   4 Aug 2017  Replace hand-written RK with scipy's RK

import numpy as np
from scipy.integrate import ode

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

def pyode45(f, dt, n_time, y0, *args):
    step = ode(f)
    step.set_integrator('dopri5')
    step.set_initial_value(y0)
    if args:
        step.set_f_params(*args)
    y = np.zeros((n_time+1, len(y0)))
    y[0,:] = y0
    t1 = n_time * dt
    i_time = 1
    while step.successful() and i_time < n_time+1:
        step.integrate(step.t + dt)
        y[i_time, :] = step.y
        i_time += 1
    assert(step.t <= t1 and step.t > t1 - dt*2)
    return y

dct_generators = {
    'euler': euler_generator
}

dct_integrators = {
    'euler': euler,
    'pyode45': pyode45
}
