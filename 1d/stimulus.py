#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2017 Sep 5  Begin

def make_square_pulse(*, space, duration, width, strength):
    input_slice = space.central_window(n_space, input_width)
    blank_input = fn_expand_param_in_population(np.zeros(n_space))
    one_pop_stim =  np.zeros(n_space)
    one_pop_stim[input_slice] = one_pop_stim[input_slice] + input_strength
    stimulus_input = fn_expand_param_in_population(one_pop_stim)

    def fn_input(t):
        if t <= duration*dt:
            return stimulus_input
        else:
            return blank_input
    return fn_input
