#!/usr/bin/env python3

import random
from copy import deepcopy
import networkx as nx
from pyvis.network import Network

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


def remove_repeated_abstract_transitions(trace: List[Transition]):
    l = []
    if not trace:
        return l
    prev_state = None
    for e, s, a, e_n, s_n in trace:
        if not prev_state or prev_state != (s, a, s_n):
            l.append((e, s, a, e_n, s_n))
        prev_state = (e, s, a, e_n, s_n)
    return l


def get_stem_and_loop(trace: List[Transition]):
    hashes = [str(e) for e, _, _, _, _ in trace]
    hashes += [str(trace[-1][3])]
    for i, x in enumerate(hashes):
        try:
            idx = hashes[i + 1:].index(x) + i + 1
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

    def get_stem(self) -> List[Transition]:
        return self.stem

    def get_loop(self) -> List[Transition]:
        return self.loop

    def get_abstract_trace(self) -> List[Transition]:
        return self.trace

    def get_abstract_stem(self) -> List[Transition]:
        return self.abstract_stem

    def get_abstract_loop(self) -> List[Transition]:
        return self.abstract_loop


class TransitionGraph(object):
    def __init__(self, env_name) -> None:
        self.env_name = env_name
        self.headers = header_register[env_name]
        self.graph = nx.MultiDiGraph()

    def add_transition(self, transition: tuple, type: str) -> None:
        _, state, act, _, next_state = transition
        s = state_to_bitstring(state)
        n_s = state_to_bitstring(next_state)

        if s not in self.graph:
            self.graph.add_node(s, title=state_to_string(state))

        if n_s not in self.graph:
            self.graph.add_node(n_s, title=state_to_string(next_state))

        type_set = set([type])
        try:
            type_set |= set(self.graph.edges[s, n_s, act]["type"])
        except KeyError:
            pass
        type_set = list(type_set)
        if "demo" in type_set and "cex" in type_set:
            color = "orange"
        elif "demo" in type_set:
            color = "green"
        elif "cex" in type_set:
            color = "red"
        self.graph.add_edge(s, n_s, key=act, label=act, color=color, type=type_set)

    def add_trace(self, trace, type) -> None:
        for transition in trace:
            self.add_transition(transition, type)

    def add_traces(self, traces, type) -> None:
        for trace in traces:
            self.add_trace(trace, type)

    def show_graph(self) -> None:
        nt = Network("500px", "500px", directed=True)
        nt.from_nx(self.graph)
        nt.show("nt.html")
