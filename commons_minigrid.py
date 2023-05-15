#!/usr/bin/env python3

import re
import pickle
import random
import itertools
import subprocess
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Callable, List, Set

import z3
import networkx as nx
from pyvis.network import Network

from minigrid.core.constants import ACT_SET
from minigrid.minigrid_env import MiniGridEnv


""" Data Types """


@dataclass
class Arguments:
    env_name: str
    spec: str
    simulator_seed: int | None = None
    demo_seed: int | None = None
    max_steps: int = 100
    tile_size: int = 32
    threshold: int = 200
    show_if_unsat: bool = False
    learner_seed: int | None = None
    save_leaner_model: bool = False


Specification = str

State = MiniGridEnv
Action = str
Transition = Tuple[State, Action, State]

FeaturesKey = Tuple[bool, ...]


class Features(dict[str, bool]):
    """A mapping from Feature Names to True/False"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.key: FeaturesKey = tuple(self[k] for k in sorted(self.keys()))

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Features):
            return False
        if self.key != other.key:
            return False
        return super().__eq__(other)


Policy = Callable[[Features], Action]
Feature_Func = Callable[[State], Features]


class Decisions(object):
    def __init__(self) -> None:
        self.features_to_actions: dict[Features, Set[Action]] = {}

    def add_decision(self, feats: Features, act: Action) -> None:
        if feats not in self.features_to_actions:
            self.features_to_actions[feats] = {act}
        else:
            self.features_to_actions[feats].add(act)

    def add_decision_list(self, decision_list: List[Tuple[Features, Action]]) -> None:
        for decision in decision_list:
            self.add_decision(*decision)

    def get_decisions(self) -> List[Tuple[FeaturesKey, Action]]:
        ## Conflicting decisions possible
        ## Delegates to learner to resolve conflict
        return [
            (feats.key, act)
            for feats in self.features_to_actions
            for act in self.features_to_actions[feats]
        ]
        ## Sampling a random decision from conflicts
        return [
            (feats.id, act)
            for feats in self.features_to_actions
            for act in random.sample(self.features_to_actions[feats], 1)
        ]


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

    def get_state_sequence(self) -> List[State]:
        return [s for s, _, _ in self.trace] + [self.trace[-1][2]]


inverse = {
    Action("left"): Action("right"),
    Action("right"): Action("left"),
    Action("pickup"): Action("drop"),
    Action("drop"): Action("pickup"),
}


class Reachability(object):
    def __init__(self) -> None:
        self.reachable: dict[int, bool] = {}
        self.adj_tr: dict[int, set[int]] = {}

    def add_node(self, u: int, r: bool = False) -> None:
        if not u in self.reachable:
            self.reachable[u] = r
            self.adj_tr[u] = set()

    def _set_as_reachable(self, s: int) -> None:
        vis = set()
        queue = deque([s])
        while queue:
            u = queue.pop()
            vis.add(u)
            if self.reachable[u]:
                continue
            self.reachable[u] = True
            for v in self.adj_tr[u]:
                if v in vis:
                    continue
                queue.append(v)

    def add_edge(self, u: int, v: int) -> None:
        self.add_node(u)
        self.add_node(v)
        self.adj_tr[v].add(u)
        if self.reachable[v]:
            self._set_as_reachable(u)

    def set_reachable(self, s: int) -> None:
        self._set_as_reachable(s)

    def can_reach(self, s: int):
        return self.reachable[s]


class AbstractGraph(object):
    def __init__(self) -> None:
        self.feats_to_ids: dict[Features, int] = {}
        self.ids_to_feats: dict[int, Features] = {}
        self.graph = nx.MultiDiGraph()
        self.ids_to_untried_acts: dict[int, Set[Action]] = {}
        self.reaching = Reachability()

        self.booleans: dict[Tuple[int, Action], z3.Bool] = {}
        self.old_models: List[z3.Model] = []

    def get_z3_bool(self, u: int, act: Action):
        if (u, act) not in self.booleans:
            self.booleans[u, act] = z3.Bool(f"{u}_{act}")
        return self.booleans[u, act]

    def get_index(self, feats: Features, add: bool = True) -> int:
        if feats in self.feats_to_ids:
            return self.feats_to_ids[feats]
        if add:
            new_id = len(self.feats_to_ids)
            self.feats_to_ids[feats] = new_id
            self.ids_to_feats[new_id] = feats
            self.ids_to_untried_acts[new_id] = deepcopy(ACT_SET)

            title = "\n".join(k for k in feats if feats[k])
            self.graph.add_node(new_id, label=f"{new_id}", title=title)
            self.reaching.add_node(new_id)
            return new_id
        return -1

    def _add_edge_id(self, u_id: int, v_id: int, a: Action) -> None:
        if self.graph.has_edge(u_id, v_id, key=a):
            return
        self.graph.add_edge(u_id, v_id, key=a, label=a)

    def _update_untried_acts(self, u_id: int, a: Action) -> None:
        if u_id not in self.ids_to_untried_acts:
            self.ids_to_untried_acts[u_id] = deepcopy(ACT_SET)
        self.ids_to_untried_acts[u_id] -= {a}

    def _remove_self_loop_on_id(self, u_id: int, a: Action) -> None:
        if not self.graph.has_edge(u_id, u_id, key=a):
            return
        remove = any(
            a == a_p and v_id != u_id
            for v_id in self.graph[u_id]
            for a_p in self.graph[u_id][v_id]
        )
        if remove:
            self.graph.remove_edge(u_id, u_id, key=a)

    def add_trace(self, trace: Trace, feature_fn: Feature_Func) -> None:
        for transition in trace:
            self.add_transition(transition, feature_fn)

    def add_transition(self, transition: Transition, feature_fn: Feature_Func) -> None:
        s, a, s_p = transition
        self.add_edge(feature_fn(s), feature_fn(s_p), a, s == s_p, True)

    def add_edge(
        self,
        u: Features,
        v: Features,
        a: Action,
        is_real_loop: bool,
        inverse_semantics: bool = True,
    ) -> None:
        u_id = self.get_index(u)
        v_id = self.get_index(v)

        self._add_edge_id(u_id, v_id, a)
        if u_id != v_id or is_real_loop:
            self._update_untried_acts(u_id, a)
        self._remove_self_loop_on_id(u_id, a)
        self.reaching.add_edge(u_id, v_id)

        if inverse_semantics and a in inverse:
            inv_a = inverse[a]
            self._add_edge_id(v_id, u_id, inv_a)
            if u_id != v_id or is_real_loop:
                self._update_untried_acts(v_id, inv_a)
            self._remove_self_loop_on_id(v_id, inv_a)
            self.reaching.add_edge(v_id, u_id)

    def set_reachable(self, u: Features) -> None:
        self.reaching.set_reachable(self.feats_to_ids[u])

    def can_reach(self, u: Features) -> bool:
        return self.reaching.can_reach(self.feats_to_ids[u])

    def get_untried_acts(self, u: Features) -> Set[Action]:
        return self.ids_to_untried_acts[self.get_index(u)]

    def get_shortest_path_edges(
        self, target_feats: Features
    ) -> List[Tuple[Features, Action]]:
        source_id = self.feats_to_ids[target_feats]
        path = nx.single_source_shortest_path(self.graph.reverse(), source=source_id)
        edges = []
        for u_id in path:
            try:
                v_id = path[u_id][-2]
                act = list(self.graph[u_id][v_id].keys())[0]
                edges.append((self.ids_to_feats[u_id], act))
            except IndexError:
                assert u_id == path[u_id][0]
        return edges

    def get_edges_from_z3(self) -> List[Tuple[Features, Action]]:
        """Get decisions using z3"""
        solver: z3.Solver = z3.Solver()

        """ Encode exactly one action for each state """
        for u in self.graph:
            solver.add(exactly_one(*[self.get_z3_bool(u, act) for act in ACT_SET]))

        """ Encode no cycles in any of the decisions taken """
        for cycle in nx.cycles.simple_cycles(self.graph):
            solver.add(
                z3.Not(
                    z3.And(
                        [
                            z3.Or(
                                [self.get_z3_bool(u, act) for act in self.graph[u][v]]
                            )
                            for u, v in zip(cycle, cycle[1:] + cycle[:1])
                        ]
                    )
                )
            )

        """ Do not repeat old model """
        for model in self.old_models:
            block_model(solver, model)

        """ Solve for the edges """
        if solver.check() != z3.sat:
            raise Exception("UNSAT!")
        model = solver.model()
        decisions = []
        for u in self.graph:
            for a in ACT_SET:
                if model[self.get_z3_bool(u, a)]:
                    # print(f"Take {a} for {u}, {[k for k in self.ids_to_feats[u] if self.ids_to_feats[u][k]]}")
                    decisions.append((self.ids_to_feats[u], a))
        self.old_models.append(deepcopy(model))
        return decisions

    def get_transitions(
        self, with_dummy=True
    ) -> List[Tuple[Features, Action, Features]]:
        transitions = [
            (self.ids_to_feats[u], a, self.ids_to_feats[v])
            for u, v, a in self.graph.edges(keys=True)
        ]
        if with_dummy:
            idx_counter = max(self.ids_to_feats.keys()) + 1
            for u_id, acts in self.ids_to_untried_acts.items():
                for a in acts:
                    transitions.append(
                        (self.ids_to_feats[u_id], a, Features({"dummy": True}))
                    )
                    idx_counter += 1
        return transitions

    def get_id_transitions(self, with_dummy=True) -> List[Tuple[int, Action, int]]:
        transitions = [(u, a, v) for u, v, a in self.graph.edges(keys=True)]
        if with_dummy:
            idx_counter = max(self.ids_to_feats.keys()) + 1
            for u, acts in self.ids_to_untried_acts.items():
                for a in acts:
                    transitions.append((u, a, idx_counter))
                    idx_counter += 1
        return transitions

    def get_id_state_limits(self, with_dummy=True) -> Tuple[int, int]:
        min_idx = min(self.ids_to_feats.keys())
        max_idx = max(self.ids_to_feats.keys())
        if with_dummy:
            max_idx += sum(len(acts) for acts in self.ids_to_untried_acts.values())
        return (min_idx, max_idx)

    def get_id_labels(self) -> dict[str, Set[int]]:
        labels = {k: set() for k in self.ids_to_feats[0]}
        for uid, feats in self.ids_to_feats.items():
            for k in labels:
                if feats[k]:
                    labels[k].add(uid)
        return labels

    def show_graph(self, filename: str = "absgraph.html") -> None:
        nt = Network("500px", "500px", directed=True)
        nt.from_nx(self.graph)
        nt.show_buttons(filter_=["physics"])
        nt.show(filename)


class ConcreteGraph(object):
    def __init__(self) -> None:
        self.state_id_to_idx = {}
        self.idx_to_state_id = {}
        self.idx_to_state = {}
        self.idx_to_untried_acts = {}
        self.graph = nx.MultiDiGraph()
        self.reach = Reachability()

        self.booleans = {}
        self.old_models = []

    def get_z3_bool(self, u: int, a: Action):
        if (u, a) not in self.booleans:
            self.booleans[u, a] = z3.Bool(f"{u}_{a}")
        return self.booleans[u, a]

    def get_index(self, s: State, add: bool = True) -> int:
        ident = s.identifier()
        if ident in self.state_id_to_idx:
            return self.state_id_to_idx[ident]
        if add:
            new_idx = len(self.idx_to_state_id)
            self.idx_to_state_id[new_idx] = ident
            self.state_id_to_idx[ident] = new_idx
            self.idx_to_state[new_idx] = deepcopy(s)
            self.idx_to_untried_acts[new_idx] = deepcopy(ACT_SET)
            return new_idx
        return -1

    def _add_edge_idx(self, s_idx: int, a: Action, s_p_idx: int):
        if not self.graph.has_edge(s_idx, s_p_idx, key=a):
            self.graph.add_edge(s_idx, s_p_idx, key=a)

    def add_transition(self, transition: Tuple[State, Action, State]):
        s, a, s_p = transition
        s_idx = self.get_index(s)
        s_p_idx = self.get_index(s_p)
        self._add_edge_idx(s_idx, a, s_p_idx)
        self.idx_to_untried_acts[s_idx] -= {a}
        self.reach.add_edge(s_idx, s_p_idx)

        if a in inverse:
            inv_a = inverse[a]
            self._add_edge_idx(s_p_idx, inv_a, s_idx)
            self.idx_to_untried_acts[s_p_idx] -= {inv_a}
            self.reach.add_edge(s_p_idx, s_idx)

    def add_trace(self, trace: Trace):
        for transition in trace:
            self.add_transition(transition)

    def set_reachable(self, s: State):
        self.reach.set_reachable(self.get_index(s))

    def can_reach(self, s: State) -> bool:
        return self.reach.can_reach(self.get_index(s))

    def get_untried_acts(self, s: State) -> Set[Action]:
        return self.idx_to_untried_acts[self.get_index(s)]

    def get_edges_from_z3(self, feature_fn: Feature_Func) -> List[Tuple[Features, Action]]:
        """Get decisions using z3"""
        solver = z3.Solver()

        """ Encode exactly one action for each state """
        for u in self.graph:
            s = self.idx_to_state[u]
            solver.add(exactly_one(*[self.get_z3_bool(feature_fn(s), act) for act in ACT_SET]))

        """ Encode no cycles in any of the decisions taken """
        for cycle in nx.cycles.simple_cycles(self.graph):
            solver.add(
                z3.Not(
                    z3.And(
                        [
                            z3.Or(
                                [self.get_z3_bool(feature_fn(self.idx_to_state[u]), act) for act in self.graph[u][v]]
                            )
                            for u, v in zip(cycle, cycle[1:] + cycle[:1])
                        ]
                    )
                )
            )

        """ Do not repeat old model """
        for model in self.old_models:
            block_model(solver, model)

        """ Solve for the edges """
        if solver.check() != z3.sat:
            raise Exception("UNSAT!")
        model = solver.model()
        decisions = []
        for u in self.graph:
            for a in ACT_SET:
                if model[self.get_z3_bool(feature_fn(self.idx_to_state[u]), a)]:
                    decisions.append((feature_fn(self.idx_to_state[u]), a))
        self.old_models.append(deepcopy(model))
        return decisions


""" Functions """

DEBUG = True


def debug(*args, **kwargs):
    """Simple debugging print statement"""
    if DEBUG or kwargs["debug"]:
        print(*args, **kwargs)


def parse_args() -> Arguments:
    """Parse all arguments for experiment"""
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-name",
        help="gym environment to load",
        default="MiniGrid-Empty-Random-6x6-v0",
    )
    parser.add_argument(
        "--spec",
        type=str,
        help="specification to check",
        default='(F "is_agent_on__goal")',
    )
    parser.add_argument(
        "--simulator-seed",
        type=int,
        help="seed for the simulator",
        default=None,
    )
    parser.add_argument(
        "--demo-seed",
        type=int,
        help="seed for the demo generator",
        default=None,
    )
    parser.add_argument(
        "--learner-seed",
        type=int,
        help="seed for the learner",
        default=None,
    )
    parser.add_argument(
        "--max-steps", type=int, help="number of steps to timeout after", default=100
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--threshold",
        type=int,
        help="number of trials after which to declare safe",
        default=200,
    )
    parser.add_argument(
        "--show-if-unsat",
        help="number of trials after which to declare safe",
        action="store_true",
    )
    parser.add_argument(
        "--save-learner-model",
        help="Whether to save the learner model",
        action="store_true",
    )
    parsed_args = parser.parse_args()

    args = Arguments(
        env_name=parsed_args.env_name,
        spec=parsed_args.spec,
        simulator_seed=parsed_args.simulator_seed,
        demo_seed=parsed_args.demo_seed,
        max_steps=parsed_args.max_steps,
        tile_size=parsed_args.tile_size,
        threshold=parsed_args.threshold,
        show_if_unsat=parsed_args.show_if_unsat,
    )
    return args


def run_bash_command(bash_command: List[str]) -> str:
    """Runs a bash command given as a list of strings"""
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    return output.decode("utf-8")


def get_stem_and_loop(
    trace: List[Transition],
) -> Tuple[List[Transition], List[Transition] | None]:
    """Get stem and loop for a given trace"""
    state_ids = [s.identifier() for s, _, _ in trace] + [trace[-1][2].identifier()]
    for i, x in enumerate(state_ids):
        try:
            idx = state_ids[i + 1 :].index(x) + i + 1
            stem, loop = trace[:i], trace[i:idx]
            return stem, loop
        except ValueError:
            continue
    return trace, None


def demo_traces_to_pickle(demos: List[Trace], env_name: str) -> None:
    """Save the list of Demo traces to a pickle file"""
    with open("data/" + env_name + "-demos.pkl", "wb") as f:
        pickle.dump(demos, f)


def pickle_to_demo_traces(env_name: str) -> List[Trace]:
    """Load a list of Demo traces from a pickle file"""
    with open("data/" + env_name + "-demos.pkl", "rb") as f:
        positive_demos = pickle.load(f)
    return positive_demos


satisfying_exp = re.compile(
    r"yes = (?P<yes>[-+]?\d+), no = (?P<no>[-+]?\d+), maybe = (?P<maybe>[-+]?\d+)"
)


def parse_number_of_reachable_states(output: str):
    x = satisfying_exp.findall(output)
    return {k: int(v) for k, v in zip(["yes", "no", "maybe"], x[0])}


state_exp = re.compile(r"(?P<pmcid>[-+]?\d+)\:\((?P<idx>[-+]?\d+)\)")


def parse_state_file(state_file: str) -> dict[int, int]:
    pmcid_idx_map = {}
    with open(state_file, "r") as f:
        _ = f.readline()
        for line in f:
            d = state_exp.match(line)
            pmcid_idx_map[int(d["pmcid"])] = int(d["idx"])
    return pmcid_idx_map


transition_exp = re.compile(
    r"(?P<upmcid>[-+]?\d+) (?P<choice>[-+]?\d+) (?P<vpmcid>[-+]?\d+) (?P<prob>[-+]?\d+) (?P<act>\S+)"
)


def parse_adv_file(adv_file: str) -> dict[int, Action]:
    pmcid_action_map = {}
    with open(adv_file, "r") as f:
        _ = f.readline()
        for line in f:
            d = transition_exp.match(line)
            pmcid_action_map[int(d["upmcid"])] = Action(d["act"])
    return pmcid_action_map


def intervals_extract(iterable):
    iterable = sorted(set(iterable))
    for _, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


def get_system(graph: AbstractGraph, with_dummy=True) -> List[str]:
    state_min, dummy_max = graph.get_id_state_limits(with_dummy=with_dummy)
    if dummy_max == state_min:
        dummy_max += 1
    state_lines = ["// World State", f"state: [{state_min}..{dummy_max}];", ""]

    transitions_lines = [
        f"[{a}] (state={u}) -> (state'={v}) ;"
        for u, a, v in graph.get_id_transitions(with_dummy=with_dummy)
    ] + [""]

    _, state_max = graph.get_id_state_limits(with_dummy=False)
    init_lines = [f"init state<={state_max} endinit", ""]

    labels = graph.get_id_labels()
    label_lines = []
    for l in labels:
        if labels[l]:
            sat_states = intervals_extract(labels[l])
            sat_states = " | ".join(
                f"(state={s1})" if s1 == s2 else f"(state>={s1} & state<={s2})"
                for s1, s2 in sat_states
            )
        else:
            sat_states = "false"
        feat = f'label "{l}" = ' + sat_states + ";"
        label_lines.append(feat)
    if with_dummy:
        label_lines.append(f'label "dummy" = (state>{state_max}) ;')
    label_lines = label_lines + [""]

    lines = (
        ["mdp", ""]
        + ["module System", ""]
        + state_lines
        + transitions_lines
        + ["endmodule", ""]
        + init_lines
        + label_lines
    )
    return [l + "\n" for l in lines]


def get_decisions(graph: AbstractGraph, spec: Specification) -> Decisions:
    lines = get_system(graph, with_dummy=False)
    with open("partmodel.prism", "w") as f:
        f.writelines(lines)
    output = run_bash_command(
        [
            "prism",
            "partmodel.prism",
            "--pf",
            f"Pmax=? [ ({spec}) ]",
            "--exportresults",
            "stdout:comment",
            "--exportadvmdp",
            "adv.tra",
            "--exportstates",
            "states.tra",
            "--exporttrans",
            "trans.tra",
        ]
    )
    reachable = parse_number_of_reachable_states(output)
    if reachable["no"] > 0:
        _ = run_bash_command(["rm", "partmodel.prism"])
        lines = get_system(graph, with_dummy=True)
        with open("partmodeldummy.prism", "w") as f:
            f.writelines(lines)
        output = run_bash_command(
            [
                "prism",
                "partmodeldummy.prism",
                "--pf",
                f'Pmax=? [ ({spec}) | (F "dummy")]',
                "--exportresults",
                "stdout:comment",
                "--exportadvmdp",
                "adv.tra",
                "--exportstates",
                "states.tra",
                "--exporttrans",
                "trans.tra",
            ]
        )
    ## Check error in output here

    pmcid_id_map = parse_state_file("states.tra")
    pmcid_action_map = parse_adv_file("adv.tra")

    # _ = run_bash_command(
    #     ["rm", "states.tra", "adv.tra", "trans.tra", "partmodel.prism"]
    # )
    decisions = Decisions()
    for pmcid, act in pmcid_action_map.items():
        id = pmcid_id_map[pmcid]
        decisions.add_decision(graph.ids_to_feats[id], act)
    return decisions


def get_decisions_reachability(graph: ConcreteGraph, feature_fn: Feature_Func):
    edges = graph.get_edges_from_z3(feature_fn)
    decisions = Decisions()
    for fs, a in edges:
        decisions.add_decision(fs, a)
    return decisions


""" z3 helper functions """


def exactly_one(*args):
    return z3.And(z3.AtMost(*args, 1), z3.Or(*args))


def block_model(solver, model):
    solver.add(z3.Or([f() != model[f] for f in model.decls() if f.arity() == 0]))
