#!/usr/bin/env python3

from typing import Tuple, Callable, List

from minigrid.minigrid_env import MiniGridEnv

State = MiniGridEnv
Features = dict[str, bool]
Action = str
Transition = Tuple[State, Action, State]
## Add Index and IndexTransition

Policy = Callable[[Features], Action]
Feature_Func = Callable[[State], Features]
Specification = str


def get_stem_and_loop(
    trace: List[Transition],
) -> Tuple[List[Transition], List[Transition] | None]:
    state_ids = [s.identifier() for s, _, _ in trace] + [trace[-1][2].identifier()]
    for i, x in enumerate(state_ids):
        try:
            idx = state_ids[i + 1 :].index(x) + i + 1
            stem, loop = trace[:i], trace[i:idx]
            return stem, loop
        except ValueError:
            continue
    return trace, None


class Trace(object):
    def __init__(self, trace: List[Transition]) -> None:
        self.trace = trace
        stem, loop = get_stem_and_loop(self.trace)
        self.stem = stem
        self.loop = loop

    def __len__(self) -> int:
        return len(self.trace)

    def __getitem__(self, index) -> Transition:
        return self.trace[index]

    def get_trace(self) -> List[Transition]:
        return self.trace

    def get_stem(self) -> List[Transition]:
        return self.stem

    def get_loop(self) -> List[Transition]:
        return self.loop

    def satisfies(self, spec: Specification, feature_fn: Feature_Func) -> bool:
        return True
