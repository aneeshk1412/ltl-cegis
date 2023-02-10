#!/usr/bin/env python3

import random
from collections import defaultdict
from pyvis.network import Network
import networkx as nx

from minigrid.core.constants import ACT_KEY_TO_IDX
from dsl_minigrid import header_register


def state_to_str(state):
    return "".join(str(int(s)) for s in state)


def str_to_state(s):
    return tuple(c == '1' for c in s)


class TransitionGraph(object):
    def __init__(self, env_name) -> None:
        self.states_to_envs = defaultdict(set)  # s -> set(e)
        self.transitions = defaultdict(
            lambda: defaultdict(set))  # s -> a -> set(s)
        self.env_name = env_name
        self.headers = header_register[env_name]
        self.nx_graph = nx.MultiDiGraph()
        self.decided_states = set()

    def add_transition(self, transition: tuple, type: str) -> None:
        env, state, act, next_env, next_state = transition
        s = state_to_str(state)
        n_s = state_to_str(next_state)

        if type == 'demo':
            self.decided_states.add(s)

        self.states_to_envs[state].add(env)
        if s not in self.nx_graph:
            self.nx_graph.add_node(
                s,
                title="\n".join(self.headers[i]
                                for i in range(len(state)) if state[i]),
            )

        self.states_to_envs[next_state].add(next_env)
        if n_s not in self.nx_graph:
            self.nx_graph.add_node(
                n_s,
                title="\n".join(
                    self.headers[i] for i in range(len(next_state)) if next_state[i]
                ),
            )

        self.transitions[state][act].add(next_state)
        type_set = set([type])
        try:
            type_set |= set(self.nx_graph.edges[s, n_s, act]['type'])
        except KeyError:
            pass
        type_set = list(type_set)
        if 'demo' in type_set and 'cex' in type_set:
            color = 'orange'
        elif 'demo' in type_set:
            color = 'green'
        elif 'cex' in type_set:
            color = 'red'
        self.nx_graph.add_edge(s, n_s, key=act, label=act,
                               color=color, type=type_set)

    def add_trace(self, trace, type) -> None:
        for transition in trace:
            self.add_transition(transition, type)

    def add_traces(self, traces, type) -> None:
        for trace in traces:
            self.add_trace(trace, type)

    def show_graph(self) -> None:
        nt = Network("500px", "500px", directed=True)
        nt.from_nx(self.nx_graph)
        nt.show("nt.html")

    def suggest_invariant_corrections(self, speculated_samples, tried_actions_for_states):
        """ Find all states which have no outgoing transitions other than self loops
            and pick a new untried action for them.
        """
        speculate_states = set()
        for s, nbrsdict in self.nx_graph.adjacency():
            if len(nbrsdict) == 1 and s in nbrsdict and s not in self.decided_states:
                state = str_to_state(s)
                speculate_states.add(state)
                tried_actions_for_states[state].update(nbrsdict[s].keys())
        for state in speculate_states:
            speculated_samples[state] = random.sample(
                [a for a in ACT_KEY_TO_IDX.keys() if a not in tried_actions_for_states[state]], 1)[0]
            tried_actions_for_states[state].add(speculated_samples[state])
        return speculated_samples, tried_actions_for_states

    def suggest_reachability_corrections(self, speculated_samples, tried_actions_for_samples):
        """ Find all cycles in the graph.
            For each cycle, pick a random state that is not in the demo set.
        """
        pass
