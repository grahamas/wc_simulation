#
# Author: Graham Smith
#
# Versions:
#   8 May 2017  Begin, copied from wc1d
#

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

class Plots(object):
    def __init__(self, results, show=False, movie_params={}):
        self.results = results
        self.show = show
        self.space_max = self.results.space_max
        self.time_max = self.results.time_max
        self.movie_params = movie_params

    def savefig(self, name, subdir=''):
        plt.savefig(self.results.pathify(name, subdir))

    def post_plot(self, *, xlabel, ylabel, title, save_to=None,
        subdir='', show=None, clear=True):
        if show is None: # Can't just "or" bc local should override
            show = self.show
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save_to:
            self.savefig(save_to, subdir)
        if show:
            plt.show()
        if clear:
            plt.clf()

    def multiline_plot(self, list_arrs, list_names, **kwargs):
        assert len(list_arrs) == len(list_names)
        for arr in list_arrs:
            plt.plot(arr)
        plt.legend(list_names)
        self.post_plot(**kwargs)

    def multiline_plot_from_dict(self, dict_arrs, **kwargs):
        list_names = dict_arrs.keys()
        list_arrs = dict_arrs.values()
        self.multiline_plot(list_arrs, list_names, **kwargs)

    def imshow(self, arr, aspect=None, **kwargs):
        if aspect is 'square':
            aspect = self.space_max / self.time_max
        plt.imshow(arr, aspect=aspect,
            extent=[0, self.space_max, 0, self.time_max])
        self.post_plot(**kwargs)

    def movie(self, movie_class, **kwargs):
        if 'save_to' in kwargs:
            kwargs['save_to'] = self.results.pathify(kwargs['save_to'])
        movie_obj = movie_class(**kwargs, **self.movie_params)
        movie_obj.run()
        movie_obj.save()

    def clear(self):
        plt.clf()


class Movie(object):
    def __init__(self, *, run=True, save_to='',
            xlabel, ylabel, title, show=False, clear=False,
            subsample=1):
        self.fig, self.ax = plt.subplots()
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.title(title)
        self._run = run
        self._save_to = save_to
        self._clear = clear
        self._subsample = subsample

    def run(self):
        self.animation = animation.FuncAnimation(self.fig, self.anim_update,
            np.arange(self.n_time), init_func=self.anim_init,
            interval=25, blit=True)

    def save(self, file_name):
        if self.animation is None:
            raise Exception("Trying to save animation before running it!")
        print(file_name)
        self.animation.save(file_name)
        if self._clear:
            plt.clf()

class WC1DMovie(Movie):
    def __init__(self, ar_activity, parse_frame=None, **kwargs):
        """
            ar_activity is assumed to be shape (n_time, n_pop, n_space).
        """
        super().__init__(**kwargs)
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
    def __init__(self, *, lines_data, **kwargs):
        super().__init__(**kwargs)
        # Remember, this subsampling makes a VIEW so don't change the data itself.
        self.lines_data = [data[::self._subsample] for data in lines_data]
        self.n_time, self.n_space = self.lines_data[0].shape
        self.x_space = np.linspace(-1, 1, self.n_space)
        assert all([(self.n_time, self.n_space) == matrix.shape
            for matrix in self.lines_data])
        self.lines = tuple(plt.plot([], [], animated=True)[0]
            for matrix in self.lines_data)
        # The below lines should be moved up to super init, but
        # I'm not sure the self.lines line can be called before self.fig
        if self._run:
            self.run()
        if self._save_to:
            self.save(self._save_to)

    def anim_init(self):
        y_max = max(map(np.max, self.lines_data))
        y_min = min(map(np.min, self.lines_data))
        x_max = 1
        x_min = -1
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        for line in self.lines:
            line.set_data(self.x_space, np.zeros_like(self.x_space))
        return self.lines

    def anim_update(self, i_frame):
        for line, matrix in zip(self.lines, self.lines_data):
            line.set_data(self.x_space, matrix[i_frame,:])
        return self.lines

