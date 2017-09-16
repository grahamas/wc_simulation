#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   7 Mar 2017  Begin
#   1 Jul 2017  Successfully replicated Neuman 2015
#   6 Jul 2017  Abstracted WC implementation to passing function to solver
#   6 Jul 2017  Made entire calculation matrix ops, no explicit populations
#   5 Sep 2017  Name changed to diffeq.py (from wc1d.py), split out some fxns

# Differences introduced from (Neuman 2015): NO LONGER TRUE?
#   Edges of weight matrix are not halved
#       (see Neuman, p98, mid-page, a.k.a. lines 39-42 of JN's script)
#   Inhibitory sigmoid function is translated to S(0) = 0

import numpy as np
import scipy

import itertools
import functools

import logging as log
log.basicConfig(level=log.INFO)

# Enable local imports from parent directory
import sys
sys.path.append("..")

# Local imports
import math_aux as math
import integrate

def beurle_mkfn(*, lattice):
    pass

def wilsoncowan73_mkfn(*, lattice, stimulus, nonlinearity, s,
    beta, alpha, r, w, tau, noise_SNR, mean_background_input, noiseless=False):
    """
        Returns a function that implements the Wilson-Cowan equations
        as parametrized in Neuman 2015.

        The RETURNED function takes the previous activity state and the
        current time index as arguments, returning the dy/dt at the
        current time index.
    """
    log.info("Making function...")
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
    log.info('Calculating connectivity...')
    mx_connectivity = calculate_connectivity_mx(dx, n_space, w, s)
    log.info('done.')
    vr_current = fn_expand_param_in_space(mean_background_input)

    log.info("Making nonlinearity...")
    fn_nonlinearity = math.dct_nonlinearities[nonlinearity['name']]
    dct_nl_args = {k: fn_expand_param_in_space(v)
        for k, v in nonlinearity['args'].items()}
    fn_transfer = functools.partial(fn_nonlinearity, **dct_nl_args)
    fn_noise = functools.partial(math.awgn,
        snr=fn_expand_param_in_space(noise_SNR))
    if not noiseless:
        fn_nonlinearity = lambda x: fn_transfer(fn_noise(x))
    else:
        fn_nonlinearity = lambda x: fn_transfer(x)

    log.info("Making input...")
    input_duration, input_strength, input_width = stimulus


    def wilsoncowan73(t, activity):
        vr_stimulus = fn_input(t)
        return (-vr_alpha * activity + (1 - activity) * vr_beta\
            * fn_nonlinearity(mx_connectivity @ activity
                + vr_current + vr_stimulus)) / vr_time_constant
    log.info("Returning function.")
    return wilsoncowan73

factories_dn = {
    'wilsoncowan73': wilsoncowan73_mkfn,
    'beurle': beurle_mkfn
}


