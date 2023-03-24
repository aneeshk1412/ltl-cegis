#!/usr/bin/env python3

from copy import deepcopy
from typing import List, Tuple

from custom_types import Transition

def remove_repeated_label_transitions(trace: List[Transition]) -> List[Transition]:
    new_trace = []
    if not trace:
        return new_trace
    prev_state = None
    for s, b, a, s_n, b_n in trace:
        if not prev_state or prev_state != (b, a, b_n):
            new_trace.append((s, b, a, s_n, b_n))
        prev_state = (s, b, a, s_n, b_n)
    return new_trace


def get_stem_and_loop(
    trace: List[Transition],
) -> Tuple[List[Transition], List[Transition] | None]:
    hashes = [hash(s) for s, _, _, _, _ in trace] + [hash(trace[-1][3])]
    for i, x in enumerate(hashes):
        try:
            idx = hashes[i + 1 :].index(x) + i + 1
            stem, loop = trace[:i], trace[i:idx]
            return stem, loop
        except ValueError:
            continue
    return trace, None


class Trace(object):
    ''' Keeps an iterator over the reduced label trace as seen by the policy
        (given the current set of features).
        Also keeps information about the original trace.
    '''
    def __init__(self, trace: List[Transition]) -> None:
        self.trace = deepcopy(trace)
        stem, loop = get_stem_and_loop(self.trace)
        self.stem = stem
        self.loop = loop
        self.label_stem = remove_repeated_label_transitions(self.stem)
        self.label_loop = remove_repeated_label_transitions(self.loop)
        self.label_trace = self.label_stem + self.label_loop

    def __len__(self) -> int:
        return len(self.label_trace)

    def __getitem__(self, index) -> Transition:
        return self.label_trace[index]

    def get_trace(self) -> List[Transition]:
        return self.trace

    def get_stem(self) -> List[Transition]:
        return self.stem

    def get_loop(self) -> List[Transition]:
        return self.loop

    def get_label_trace(self) -> List[Transition]:
        return self.label_trace

    def get_label_stem(self) -> List[Transition]:
        return self.label_stem

    def get_label_loop(self) -> List[Transition]:
        return self.label_loop
