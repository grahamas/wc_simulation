#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2017 Sep 6  Begin
#

import numpy as np

from cached_property import cached_property

class Lattice(object):
    def __init__(self, *, space_extent, space_step,
        time_extent, time_step,
        n_populations, population_names):
        '''
            Args:
                space_extent    : Dimensional size of space dimension
                space_step      : Step size of space, i.e. dx
                time_extent     : Dimensional size of time dimension
                time_step       : Step size of time, i.e. dt
                n_populations   : Number of neural populations
                population_names: Names of populations
            Returns:
                Lattice of size space_extent x time_extent x n_populations
        '''
        self.space_extent = space_extent
        self.space_step = space_step
        self.time_extent = time_extent
        self.time_step = time_step
        self.n_populations = n_populations
        self.n_space = self.nondimensionalize(space_extent, space_step)
        assert isinstance(self.n_space, int)
        self.n_time = self.nondimensionalize(time_extent, time_step)
        self.population_names = population_names
        self.shape = (self.n_time, self.n_populations, self.n_space)
        self.sim_shape = (self.n_time+1, self.n_populations, self.n_space)
    def nondimensionalize(self, extent, step):
        '''
            Translates from dimensional terms to non-dimensional array size

            Args:
                extent: Dimensioned length
                step: Dimensioned step size
            Returns:
                Number of points to represent extent with step stepsize
        '''
        return int(extent / step)
    def space_frame(self):
        return np.empty((self.n_space * self.n_populations,))
    @cached_property
    def array(self):
        # It's possible caching this is worse than the if statement?
        # But so clean.
        self._array = np.empty((self.n_time, self.n_space, self.n_populations))
        return self._array
    def expand_in_space(self, vec):
        return np.repeat(vec, self.n_space)
    def expand_in_population(self, vec):
        return np.tile(vec, self.n_populations)
    def central_window(self, width):
        """
            Calculate indices for a region of width window_width roughly in the
            center of an array of length total_len.

            window_width is DIMENSIONED
        """
        n_width = self.nondimensionalize(width, self.space_step)
        median_dx = int(self.n_space // 2)
        half_width = int(n_width // 2)
        assert half_width < median_dx
        if n_width % 2: # is odd
            window_slice = slice(median_dx - half_width, median_dx + half_width + 1)
        else:
            window_slice = slice(median_dx - half_width, median_dx + half_width)
        return window_slice

# class Lattice1D(Lattice1):
#     '''
#         Contains space or time variables for ease of access.
#     '''
#     def __init__(self, *, step, length):
#         '''
#             Arguments are dimensional.



# class Lattice0(Lattice):
#     '''
#         A zero-dimensional lattice (i.e. a point).
#         Really it's a set, but that's already a class and I don't want
#         to have my larger lattice be heterogenous.
#     '''
#     def __init__(self, *, n_points):
#         self.n_points = n_points

# class Lattice1(Lattice):
#     '''
#         A latice in one dimension, but without dimension.
#     '''
#     def __init__(self, *, n_points):
#         self.n_points = n_points
#     def central_window(self, window_width):
#         """
#             Calculate indices for a region of width window_width roughly in the
#             center of an array of length total_len.
#         """
#         median_dx = self.n_points // 2
#         half_width = window_width // 2
#         assert half_width < median_dx
#         if window_width % 2: # is odd
#             window_slice = slice(median_dx - half_width, median_dx + half_width + 1)
#         else:
#             window_slice = slice(median_dx - half_width, median_dx + half_width)
#         assert len(window_slice) == window_width
#         return window_slice

# class Lattice1D(Lattice1):
#     '''
#         Contains space or time variables for ease of access.
#     '''
#     def __init__(self, *, step, length):
#         '''
#             Arguments are dimensional.

#             Step is dx, end the mesh's length.
#         '''
#         self.step = step
#         self.length = length
#         super.__init__(n_points=int(self.length / self.step))
