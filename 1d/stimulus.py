#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2017 Sep 5  Begin

import numpy as np

def stimulus_mkfn(*, lattice, name, args):
    factory = stimulus_dn[name]
    return factory(lattice=lattice, **args)
    

def square_pulse_mkfn(*, lattice, duration, width, strength):
    n_space, dt = lattice.n_space, lattice.space_step
    blank_stim = lattice.expand_in_population(np.zeros(n_space))
    one_pop_stim =  np.zeros(n_space)
    stim_slice = lattice.central_window(width)
    assert len(one_pop_stim[stim_slice]) == lattice.nondimensionalize(width, lattice.space_step),\
            "{} not length {}".format(stim_slice, width)
    one_pop_stim[stim_slice] = one_pop_stim[stim_slice] + strength
    stim = lattice.expand_in_population(one_pop_stim)

    def fn_stim(t):
        if t <= duration*dt:
            return stim
        else:
            return blank_stim
    return fn_stim

stimulus_dn = {
        "square_pulse": square_pulse_mkfn
        }
