#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2017 Sep 5  Split from diffeq.py (formerly wc1d.py)
#

import logging as log
log.basicConfig(level=log.INFO)

# Enable local imports from parent directory
import sys
sys.path.append("..")

# Local imports
from plot import LinesMovieFromSepPops, ResultInfo, ResultPlots
from analyse import TimeSpace
from args import load_args_from_file

@load_args_from_file
def run_simulation(modifications=None, show_figs=False,
    movie_params=None, show_timespace=None, do_analysis=False, **model_params):
    """
        Run the simulation resulting from simulate_neuman, and
        save results.
    """
    if show_timespace is None:
        show_timespace = show_figs
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
        clear=(not show_timespace))
    timespace_E = TimeSpace(E)
    if do_analysis:
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
