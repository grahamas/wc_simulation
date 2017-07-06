#
# Author: Graham Smith
#
# Versions:
#   8 May 2017  Begin, copied from wc1d
#

import time, os
import json

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

class PlotDirectory(object):

    def __init__(self, name, params, sep='_', root='plots'):
        self._base_name = name
        self._init_time = time.strftime("%Y%m%d{}%H%M%S".format(sep))
        dir_name = self._init_time + sep + name
        self._dir_path = os.path.join(root, dir_name)
        if os.path.exists(self._dir_path):
            raise Exception('"Uniquely" named folder already exists.')
        os.mkdir(self._dir_path)
        with open(self.pathify('params.json'), 'w') as jsonfile:
            json.dump(params, jsonfile)

    def pathify(self, name, subdir=''):
        return os.path.join(self._dir_path, subdir, name)

    def savefig(self, name, subdir=''):
        plt.savefig(self.pathify(name, subdir))


class Movie(object):
    def run(self):
        self.animation = animation.FuncAnimation(self.fig, self.anim_update,
            np.arange(self.n_time), init_func=self.anim_init,
            interval=25, blit=True)

    def save(self, file_name):
        # TODO: Should catch hasn't run yet
        print(file_name)
        self.animation.save(file_name)

class WC1DMovie(Movie):
    def __init__(self, ar_activity, run=True, save_to='',
        parse_frame=None):
        """
            ar_activity is assumed to be shape (n_time, n_pop, n_space).
        """
        self.fig, self.ax = plt.subplots()
        self.ar_activity = ar_activity
        self.n_time, self.n_population, self.n_space = ar_activity.shape
        self.x_space = np.linspace(-1, 1, self.n_space)
        if parse_frame:
            self.parse_frame = parse_frame
        else:
            self.parse_frame = self.default_parse_frame
        self.n_lines = len([x for x in parse_frame(ar_activity[0,:,:])])
        self.lines = tuple(plt.plot([], [], animated=True)[0]
            for i_line in range(self.n_lines))

    def default_parse_frame(self, frame):
        for i_pop in range(self.n_population):
            yield frame[i_pop,:]
        yield np.sum(frame,axis=1)

class LinesMovieFromSepPops(Movie):
    def __init__(self, matrices, run=True, save_to=''):
        self.fig, self.ax = plt.subplots()
        self.matrices = matrices
        self.n_time, self.n_space = self.matrices[0].shape
        self.x_space = np.linspace(-1, 1, self.n_space)
        assert all([(self.n_time, self.n_space) == matrix.shape for matrix in self.matrices])
        self.lines = tuple(plt.plot([], [], animated=True)[0]
            for matrix in self.matrices)
        if run:
            self.run()
        if save_to:
            self.save(save_to)

    def anim_init(self):
        y_max = max(map(np.max, self.matrices))
        y_min = min(map(np.min, self.matrices))
        x_max = 1
        x_min = -1
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        for line in self.lines:
            line.set_data(self.x_space, np.zeros_like(self.x_space))
        return self.lines

    def anim_update(self, i_frame):
        for line, matrix in zip(self.lines, self.matrices):
            line.set_data(self.x_space, matrix[i_frame,:])
        return self.lines

