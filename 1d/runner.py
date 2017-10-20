#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2017 Sep 5  Split from diffeq.py (formerly wc1d.py)
#

import logging as log
log.basicConfig(level=log.INFO)

from multiprocessing import Pool
from itertools import islice

# Enable local imports from parent directory
import sys
sys.path.append("..")

# Local imports
from analyse import Results
from args import load_args_from_file, inner_dcts_pop, Parametrizations
from lattice import Lattice
from diffeq import factories_dn
import integrate

def simulate(*, lattice, solver, equation_name, **params):
    """
        Simulates the a differential equation made from a given factory.
    """
    log.info("In simulation function.")
    lattice = Lattice(**lattice)

    factory_fn = factories_dn[equation_name]
    diffeq_fn = factory_fn(lattice=lattice, **params)

    log.info('starting simulation...')
    initial_value = lattice.space_frame()
    simulation_step = lattice.time_step
    n_simulation_length = lattice.n_time
    solver_fn = integrate.dct_integrators[solver["name"]]
    activity = solver_fn(diffeq_fn, simulation_step, n_simulation_length,
        initial_value)\
        .reshape(lattice.sim_shape)
    log.info('simulation done.')

    return activity

def sim_and_analyse(model_params):
    run_name = model_params.pop('run_name')
    results_params = model_params.pop('results_params')
    activity = simulate(**model_params)
    results = Results(data=activity, **results_params,
            run_name=run_name, model_params=model_params)
    results.analyse()
    log.info("All done with {}.".format(run_name))

@load_args_from_file()
def run_simulation(*, model_modifications, parent_dir=None, **base_model_params):
    """
        Run the simulation resulting from simulate_neuman, and
        save results.
    """
    log.info("Starting parallel simulations...")

    model_params_itr = Parametrizations({**base_model_params,
        **model_modifications})

    ### TESTING ###
    log.info("Initializing pool...")
    p = Pool(12)
    log.info("Pool initialized.")
    p.map(sim_and_analyse, islice(model_params_itr, 0, 10))