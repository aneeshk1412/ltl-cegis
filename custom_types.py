#!/usr/bin/env python3
import networkx as nx

class PartialMDP(object):
    def __init__(self, obj_to_state_fn) -> None:
        self.objs_to_idxs = {'dummy': 0}
        self.idxs_to_objs = {0: 'dummy'}
        self.objs_to_state = {'dummy': None}
        self.obj_to_state_fn = obj_to_state_fn
        self.graph = nx.DiGraph()

    def add_state_and_get_index(self, state) -> int:
        obj = self.obj_to_state_fn(state)
        if obj not in self.objs_to_idxs:
            new_idx = len(self.objs_to_idxs)
            self.objs_to_idxs[obj] = new_idx
            self.idxs_to_objs[new_idx] = obj
            self.objs_to_state[obj] = state
        return self.objs_to_idxs[obj]

    def add_trace(self, trace):
        self.graph.add_edges_from([(self.add_state_and_get_index(s), self.add_state_and_get_index(s_p), {'action': a}) for s, _, a, s_p, _ in trace])

    def print_prism(self):
        for u, v, act in self.graph.edges(data='action'):
            print(f"[{act}] (state={u}) -> (state={v})")

