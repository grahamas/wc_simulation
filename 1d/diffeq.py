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
from stimulus import stimulus_mkfn

def beurle_mkfn(*, lattice, stimulus, m):
    """
        Returns a function that implements the second-order ODE found in
        Beurle, 1956 for the proportion of sensitive cells in a population.

        Specifically: R_tt = m R R_t - R_t
        is transformed into the first order system
        x_2' = m x_1 x_2 - x_2
        x_1' = x_2

        Where x_1' = x_2 = F is the quantity we relate to the WC equations,
        namely the rate at which cells become sensitive, which is, according
        to Beurle, the negative of the rate at which cells become active.
    """
    # TODO: incoporate stimulus

    def beurle56(t, activity):
        new_activity = np.array([activity[1], 
            m * activity[0] * activity[1] - activity[1]])
        return new_activity
    return beurle56

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
    n_space, dx = lattice.n_space, lattice.space_step
    n_time, dt = lattice.n_time, lattice.time_step
    n_population = lattice.n_populations
    # TODO: native use of lattice

    fn_expand_param_in_space = lambda x: np.repeat(x, n_space)
    fn_expand_param_in_population = lambda x: np.tile(x, n_population)

    vr_time_constant = fn_expand_param_in_space(tau)
    vr_alpha = fn_expand_param_in_space(alpha)
    vr_beta = fn_expand_param_in_space(beta)
    log.warn('wilsoncowan73 refraction ignored; r=1 assumed.')
    log.info('Calculating connectivity...')
    mx_connectivity = \
            math.sholl_1d_connectivity_mx(lattice, w, s)
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
    fn_forcing = stimulus_mkfn(lattice=lattice, **stimulus)

    def wilsoncowan73(t, activity):
        vr_stimulus = fn_forcing(t)
        return (-vr_alpha * activity + (1 - activity) * vr_beta\
            * fn_nonlinearity(mx_connectivity @ activity
                + vr_current + vr_stimulus)) / vr_time_constant
    log.info("Returning function.")
    return wilsoncowan73

factories_dn = {
    'wilsoncowan73': wilsoncowan73_mkfn,
    'beurle': beurle_mkfn
}