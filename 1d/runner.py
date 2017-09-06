#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2017 Sep 5  Split from diffeq.py (formerly wc1d.py)
#

import json
import time
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import logging as log
log.basicConfig(level=log.INFO)

# Enable local imports from parent directory
import sys
sys.path.append("..")

# Local imports
from plot import LinesMovieFromSepPops, ResultInfo, ResultPlots
from analyse import TimeSpace

#region json helpers
def read_jsonfile(filename):
    """Reads jsonfile of name filename and returns contents as dict."""
    with open(filename, 'r') as f:
        params = json.load(f)
    return params
def load_json_with(loader, pass_name=False):
    """
        A decorator for functions whose first argument is a dict
        that could be loaded from a json-file.

        If the first argument is a string rather than a dict, the wrapper
        tries to load a file with that string name and then pass the
        loaded dict (along with the filename, optionally) to the
        wrapped function.
    """
    def decorator(func=read_jsonfile):
        def wrapper(*args, **kwargs):
            if isinstance(args[0], str):
                loaded = loader(args[0])
                if pass_name:
                    run_name = os.path.splitext(args[0])[0]
                    out_args = [loaded, run_name, *args[1:]]
                else:
                    out_args = [loaded, *args[1:]]
                return func(*out_args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
def get_neuman_params_from_json(json_filename, params_dir='params'):
    params = read_jsonfile(os.path.join(params_dir,json_filename))
    run_name = os.path.splitext(json_filename)[0]
    params['r'] = [1,1]
    log.warn('WARNING: Using Neuman equations without refraction.')
    return params
#endregion

@load_json_with(get_neuman_params_from_json, pass_name=True)
def run_simulation(params, run_name, modifications=None, show_figs=False,
    movie_params=None, timespace_show=None, analysis=True):
    """
        Run the simulation resulting from simulate_neuman, and
        save results.
    """
    if timespace_show is None:
        timespace_show = show_figs
    if modifications:
        for key, value in modifications.items():
            params[key] = value

    activity = simulate_neuman(**params)
    E = activity[:,0,:]
    I = activity[:,1,:]
    result_info = ResultInfo(run_name, params)
    result_plots = ResultPlots(result_info, show=show_figs)
    plt.clf()
    if movie_params:
        result_plots.movie(movie_class=LinesMovieFromSepPops,
                lines_data=[E, I, E+I],
                save_to='activity_movie.mp4',
                xlabel='space (a.u.)', ylabel='amplitude (a.u.)',
                title='1D Wilson-Cowan Simulation', **movie_params)
    result_plots.imshow(E, xlabel='space (a.u.)', ylabel='time (a.u.)',
        title='Heatmap of E activity', save_to='E_timespace.png',
        clear=(not timespace_show))
    timespace_E = TimeSpace(E)
    if analysis:
        bump_properties = timespace_E.clooge_bump_properties()
        peak_ix = bump_properties.pop('peak_ix')
        width = bump_properties.pop('width')
        amplitude = bump_properties.pop('amplitude')
        result_plots.multiline_plot([amplitude], ['amplitude'], xlabel='time (a.u.)',
            ylabel='(a.u., various)', title='Properties of moving bump',
            save_to='cloogy_bump_amp.png')
        result_plots.multiline_plot_from_dict(bump_properties, xlabel='time (a.u.)',
            ylabel='(a.u., various)', title='Properties of moving bump',
            save_to='cloogy_bump_vel.png')
        result_plots.multiline_plot([width], ['width'], xlabel='time (a.u.)',
            ylabel='(a.u., various)', title='Width of moving bump',
            save_to='cloogy_bump_width.png')
        result_info.save_data(data=bump_properties, filename='bump_properties.pkl')
