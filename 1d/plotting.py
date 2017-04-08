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


class LineMovie(object):
    def __init__(self, matrix, run=True, save_to=''):
        self.matrix = matrix
        self.n_data, self.n_time = self.matrix.shape

        self.fig, self.ax = plt.subplots()
        self.x_data = np.linspace(-1, 1, self.n_data)
        self.line, = plt.plot([], [], 'ro', animated=True)
        if run:
            self.run()
        if save_to:
            self.save(save_to)

    def anim_init(self):
        y_max = np.max(self.matrix)
        y_min = np.min(self.matrix)
        x_max = 1
        x_min = -1
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.line.set_data(self.x_data, np.zeros_like(self.x_data))
        return self.line,

    def anim_update(self, i_frame):
        self.line.set_data(self.x_data, self.matrix[:,i_frame])
        return self.line,

    def run(self):
        self.animation = animation.FuncAnimation(self.fig, self.anim_update, 
            np.arange(self.n_time), init_func=self.anim_init,
            interval=25, blit=True)

    def save(self, file_name):
        # TODO: Should catch hasn't run yet
        print(file_name)
        self.animation.save(file_name)