#!/usr/bin/env python3

#
# Author: Graham Smith
#
# Versions:
#   2017 Sep 6  Split from runner.py
#

import json
import os
from functools import reduce
import itertools
from copy import deepcopy

#region json helpers
def read_json_file(filename):
    """Reads jsonfile of name filename and returns contents as dict."""
    with open(filename, 'r') as f:
        params = json.load(f)
    return params
def load_args_from_file(loader=read_json_file):
    """
        A decorator for functions whose arguments include keyword arguments
        that could at least in part be loaded from a json file.
    """
    def decorator(func):
        def wrapper(*args, json_file_name=None, **kwargs):
            if json_file_name:
                if 'json_dir' in kwargs:
                    file_path = os.path.join(kwargs.pop('json_dir'),json_file_name)
                else:
                    file_path = json_file_name
                loaded = loader(file_path)
                if not ('run_name' in kwargs or 'run_name' in loaded):
                    run_name = os.path.splitext(args[0])[0]
                    loaded['run_name'] = json_file_name
                return func(*args, **kwargs, **loaded)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

#endregion


def add_dcts (dct1, dct2):
    return {**dct1, **dct2}

def multikey(dct, l_keys):
    return [dct[key] for key in l_keys]
def inner_dcts(dct, l_keys):
    '''
        Extracts and combines dictionaries inside dct keyed by l_keys

        Args:
            l_keys: list of keys
        Returns:
            Dictionary containing union of all dictionaries of the form
            dct[key] for key in l_keys.
    '''
    return reduce(add_dcts, multikey(dct, l_keys))

def multikey_pop(dct, l_keys):
    return [dct.pop(key) for key in l_keys]
def inner_dcts_pop(dct, l_keys):
    '''
        Extracts and combines dictionaries inside dct keyed by l_keys

        Args:
            l_keys: list of keys
        Returns:
            Dictionary containing union of all dictionaries of the form
            dct[key] for key in l_keys.
    '''
    return reduce(add_dcts, multikey(dct, l_keys))

class ParameterSet(set):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

class Parametrizations(object):
    '''
        Takes dictionary of params (param_d) that possibly has sets
        in place of parameters, and returns back the "product" of
        parametrizations wrt those sets.

        NOT PARALLEL!
    '''
    def __init__(self, param_d):
        self.param_d = param_d
        self.base_run_name = param_d['run_name']
        self.set_path_l = []
        self.set_name_l = []
        self.set_l = []
        self.find_all_set_paths(self.param_d, [])
        self.set_product = itertools.product(*self.set_l)
        self.cleanse_param_d()

    def __iter__(self):
        return self

    def __next__(self):
        param_changes = next(self.set_product)
        return self.new_param_d(param_changes)

    def find_all_set_paths(self, level, this_path):
        if isinstance(level, ParameterSet):
            self.set_path_l += [this_path]
            self.set_name_l += [level.name]
            self.set_l += [level]
            return
        if isinstance(level, dict):
            for key, value in level.items():
                self.find_all_set_paths(value, this_path + [key])
        elif isinstance(level, list):
            for idx, value in enumerate(level):
                self.find_all_set_paths(value, this_path + [idx])
        return

    def new_param_d(self, param_changes):
        param_d_copy = deepcopy(self.param_d)
        self.set_sets(param_d_copy, param_changes)
        param_changes_str = map(str, param_changes)
        mod_str = '_'.join([''.join(pair)
            for pair in zip(self.set_name_l, param_changes_str)
            if pair[0]])
        param_d_copy['run_name'] = '_'.join([self.base_run_name, mod_str])
        return param_d_copy

    def set_sets(self, param_d, param_changes):
        for path, change in zip(self.set_path_l, param_changes):
            self.set_set(param_d, path, change)
    def set_set(self, param_d, path, change):
        '''
            Iterate through parameters until you reach the level
            before the change. Then exit the loop and use the idx_or_key
            to make the change. Everything is by reference, so the change
            carries over to the instance's copy.
        '''
        level = param_d
        for idx_or_key in path[:-1]:
            level = level[idx_or_key]
        level[path[-1]] = change
    def cleanse_param_d(self):
        self.set_sets(self.param_d, [None] * len(self.set_path_l))
