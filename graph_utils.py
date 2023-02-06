#!/usr/bin/env python3

from typing import List, Tuple
from minigrid.minigrid_env import MiniGridEnv

from dsl_minigrid import feature_register, header_register


def env_to_state(env: MiniGridEnv, env_name: str) -> Tuple[bool, ...]:
    return feature_register[env_name](env)


def state_to_bitstring(state: Tuple[bool, ...]) -> str:
    return "".join(int(s) for s in state)


def bitstring_to_state(s: str) -> Tuple[bool, ...]:
    return tuple(c == "1" for c in s)


def state_to_string(state: Tuple[bool, ...], env_name: str) -> str:
    return "\n".join(header_register[env_name][i] for i, s in enumerate(state) if s)


def bitstring_to_string(s: str, env_name: str) -> str:
    return "\n".join(header_register[env_name][i] for i, c in enumerate(s) if c == "1")


Transition = Tuple[MiniGridEnv, Tuple[bool, ...], str, MiniGridEnv, Tuple[bool, ...]]


class Trace(object):
    def __init__(self, trace: List[Transition], type: str) -> None:
        self.trace = trace
        self.type = type

    def __len__(self):
        return len(self.trace)

    def __getitem__(self, index):
        return self.trace[index]

    def get_loop(self):
        pass
