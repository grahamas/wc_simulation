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
