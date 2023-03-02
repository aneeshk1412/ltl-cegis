#!/usr/bin/env python3

from copy import deepcopy

from typing import List, Tuple
from minigrid.minigrid_env import MiniGridEnv


Transition = Tuple[MiniGridEnv, Tuple[bool, ...], str, MiniGridEnv, Tuple[bool, ...]]


def remove_repeated_abstract_transitions(trace: List[Transition]):
    abstract_transitions = []
    if not trace:
        return abstract_transitions
    prev_state = None
    for e, s, a, e_n, s_n in trace:
        if not prev_state or prev_state != (s, a, s_n):
            abstract_transitions.append((e, s, a, e_n, s_n))
        prev_state = (e, s, a, e_n, s_n)
    return abstract_transitions


def get_stem_and_loop(trace: List[Transition]):
    hashes = [str(e) for e, _, _, _, _ in trace] + [str(trace[-1][3])]
    for i, x in enumerate(hashes):
        try:
            idx = hashes[i + 1 :].index(x) + i + 1
            stem, loop = trace[:i], trace[i:idx]
            return stem, loop
        except ValueError:
            continue
    return trace, None


class Trace(object):
    def __init__(self, trace: List[Transition], type: str = None) -> None:
        self.type = type
        self.trace = deepcopy(trace)
        stem, loop = get_stem_and_loop(self.trace)
        self.stem = stem
        self.loop = loop
        self.abstract_stem = remove_repeated_abstract_transitions(self.stem)
        self.abstract_loop = remove_repeated_abstract_transitions(self.loop)
        self.abstract_trace = self.abstract_stem + self.abstract_loop

    def __len__(self) -> int:
        return len(self.abstract_trace)

    def __getitem__(self, index) -> Transition:
        return self.abstract_trace[index]

    def __lt__(self, other) -> bool:
        return True

    def get_stem(self) -> List[Transition]:
        return self.stem

    def get_loop(self) -> List[Transition]:
        return self.loop

    def get_abstract_trace(self) -> List[Transition]:
        return self.abstract_trace

    def get_abstract_stem(self) -> List[Transition]:
        return self.abstract_stem

    def get_abstract_loop(self) -> List[Transition]:
        return self.abstract_loop
