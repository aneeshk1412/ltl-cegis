#!/usr/bin/env python3

import networkx as nx
from typing import List
from pyvis.network import Network

from trace_minigrid import Trace, Transition
from utils import state_to_bitstring, state_to_string


class TransitionGraph(object):
    def __init__(self, env_name: str) -> None:
        self.env_name = env_name
        self.graph = nx.MultiDiGraph()

    def add_transition(self, transition: Transition, type: str, label=None) -> None:
        _, state, act, _, next_state = transition
        s = state_to_bitstring(state)
        n_s = state_to_bitstring(next_state)

        if s not in self.graph:
            self.graph.add_node(
                s,
                title=state_to_string(state, self.env_name),
                color="green" if type == "demo" else "red",
            )

        if n_s not in self.graph:
            self.graph.add_node(
                n_s,
                title=state_to_string(next_state, self.env_name),
                color="green" if type == "demo" else "red",
            )

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
        if label is None:
            self.graph.add_edge(s, n_s, key=act, label=act, color=color, type=type_set)
        else:
            self.graph.add_edge(s, n_s, key=(act, label), label=', '.join([act, label]), color=color, type=type_set)

    def add_trace(self, trace: Trace, label=None) -> None:
        for transition in trace:
            self.add_transition(transition, trace.type, label=label)

    def add_traces(self, traces: List[Trace], label=None) -> None:
        for trace in traces:
            self.add_trace(trace, label=label)

    def show_graph(self, name="nt.html") -> None:
        nt = Network("500px", "500px", directed=True)
        nt.from_nx(self.graph)
        nt.show_buttons(filter_=['physics'])
        nt.show(name)
