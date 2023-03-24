#!/usr/bin/env python3

from typing import Tuple, NewType, TypeVar, TypeAlias, Callable

""" Define Types """
State = TypeVar('State')
Action = NewType('Action', str)
Features = NewType('Features', Tuple[bool, ...])
Transition = NewType('Transition', Tuple[State, Features, Action, State, Features])
Policy: TypeAlias = Callable[[Features], Action]
