#!/usr/bin/python

import yaml
from abc import ABC, abstractmethod
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

class Terminal(ABC):
    def __init__(self, term) -> None:
        self.term = term
        if not self.__verify__():
            raise NotImplementedError
    
    def __str__(self) -> str:
        return str(self.term)

    def __cstr__(self) -> str:
        return str(self.term)

    @abstractmethod
    def __verify__(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def __simple_enumerate__(cls):
        pass


class NonTerminal(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.terms = list(args)
        if not self.__verify__():
            print(self.terms)
            raise NotImplementedError
    
    def __str__(self) -> str:
        return ''
    
    @abstractmethod
    def __cstr__(self) -> str:
        return ''

    @abstractmethod
    def __verify__(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def __simple_enumerate__(cls):
        pass


def open_config_file(configfile):
    with open(configfile, 'r') as stream:
        return yaml.safe_load(stream)