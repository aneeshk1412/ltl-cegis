#!/usr/bin/env python3

import networkx as nx
from typing import Tuple, List
from pyvis.network import Network

from minigrid.minigrid_env import MiniGridEnv

from trace_minigrid import Trace, Transition
from dsl_minigrid import feature_register, header_register


def env_to_state(env: MiniGridEnv, env_name: str) -> Tuple[bool, ...]:
    return feature_register[env_name](env)


def state_to_bitstring(state: Tuple[bool, ...]) -> str:
    return "".join(str(int(s)) for s in state)


def bitstring_to_state(s: str) -> Tuple[bool, ...]:
    return tuple(c == "1" for c in s)


def state_to_string(state: Tuple[bool, ...], env_name: str) -> str:
    return "\n".join(header_register[env_name][i] for i, s in enumerate(state) if s)


def bitstring_to_string(s: str, env_name: str) -> str:
    return "\n".join(header_register[env_name][i] for i, c in enumerate(s) if c == "1")


class TransitionGraph(object):
    def __init__(self, env_name: str) -> None:
        self.env_name = env_name
        self.graph = nx.MultiDiGraph()

    def add_transition(self, transition: Transition, type: str) -> None:
        _, state, act, _, next_state = transition
        s = state_to_bitstring(state)
        n_s = state_to_bitstring(next_state)

        if s not in self.graph:
            self.graph.add_node(s, title=state_to_string(state, self.env_name))

        if n_s not in self.graph:
            self.graph.add_node(n_s, title=state_to_string(next_state, self.env_name))

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

    def add_trace(self, trace: Trace) -> None:
        for transition in trace:
            self.add_transition(transition, trace.type)

    def add_traces(self, traces: List[Trace]) -> None:
        for trace in traces:
            self.add_trace(trace)

    def show_graph(self) -> None:
        nt = Network("500px", "500px", directed=True)
        nt.from_nx(self.graph)
        nt.show("nt.html")
