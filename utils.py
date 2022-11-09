#!/usr/bin/python3

import yaml
from itertools import count, repeat

def interleave(*iterables):
    yield from (ele for tups in zip(*iterables) for ele in tups)

def icombinations(f):
    for i, x in zip(count(), f()):
        yield from zip(f(), repeat(x, i + 1))

def iproduct(f, g):
    for i, x, y in zip(count(), f(), g()):
        yield from zip(repeat(x, i), g())
        yield from zip(f(), repeat(y, i+1))

def grouped2(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def cstr(obj):
    try:
        return obj.__cstr__()
    except AttributeError:
        return str(obj)

def open_config_file(configfile):
    with open(configfile, 'r') as stream:
        return yaml.safe_load(stream)