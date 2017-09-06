#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2017 Sep 6  Begin
#

class Lattice(object):
    def __init__(self, *, d_n_points=None, d_lattices=None):
        if d_lattices:
            self._d_lattices = d_lattices
        elif d_n_points:
            self._d_lattices = {}
            for key, value in d_n_points.items():
                if isinstance(value, dict):
                    try:
                        #TODO: This has the (as written) unfixable bug
                        # in the case where two dimensions are named
                        # delta and length
                        self._d_lattices[key] = Lattice1D(**value)
                    except TypeError:
                        self._d_lattices[key] = Lattice(d_n_points=value)
                elif isinstance(value, int):
                    self._d_lattices[key] = Lattice1(n_points=value)
        else:
            raise ValueError('Lattice only takes one of two arguments.')
    def __getitem__(self, key):
        return self._d_lattices[key]

class Lattice0(Lattice):
    '''
        A zero-dimensional lattice (i.e. a point).
        Really it's a set, but that's already a class and I don't want
        to have my larger lattice be heterogenous.
    '''
    def __init__(self, *, n_points):
        self.n_points = n_points

class Lattice1(Lattice):
    '''
        A latice in one dimension, but without dimension.
    '''
    def __init__(self, *, n_points):
        self.n_points = n_points
    def central_window(self, window_width):
        """
            Calculate indices for a region of width window_width roughly in the
            center of an array of length total_len.
        """
        median_dx = self.n_points // 2
        half_width = window_width // 2
        assert half_width < median_dx
        if window_width % 2: # is odd
            window_slice = slice(median_dx - half_width, median_dx + half_width + 1)
        else:
            window_slice = slice(median_dx - half_width, median_dx + half_width)
        assert len(window_slice) == window_width
        return window_slice

class Lattice1D(Lattice1):
    '''
        Contains space or time variables for ease of access.
    '''
    def __init__(self, *, step, length):
        '''
            Arguments are dimensional.

            Step is dx, end the mesh's length.
        '''
        self.step = step
        self.length = length
        super.__init__(n_points=int(self.length / self.step))
