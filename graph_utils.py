#!/usr/bin/env python3

import random
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


class Trace(object):
    def __init__(self, trace: List[Transition], type: str) -> None:
        self.trace = trace
        self.type = type

    def __len__(self):
        return len(self.trace)

    def __getitem__(self, index):
        return self.trace[index]

    def get_loop(self):
        stem, _ = self.get_stem_and_loop()
        return stem

    def get_loop(self):
        _, loop = self.get_stem_and_loop()
        return loop

    def get_stem_and_loop(self):
        hashes = [str(e) for e, _, _, _, _ in self.trace]
        hashes += [str(self.trace[-1][3])]
        for i, x in enumerate(hashes):
            try:
                idx = hashes[i + 1:].index(x) + i + 1
                stem, loop = self.trace[:i], self.trace[i:idx]
                return Trace(stem, self.type), Trace(loop, self.type)
            except ValueError:
                continue
        return Trace(self.trace, self.type), None


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
