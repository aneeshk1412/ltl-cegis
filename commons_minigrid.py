#!/usr/bin/env python3

import re
import pickle
import random
import itertools
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Callable, List, Set

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


inverse = {
    Action("left"): Action("right"),
    Action("right"): Action("left"),
    Action("pickup"): Action("drop"),
    Action("drop"): Action("pickup"),
}


class AbstractGraph(object):
    def __init__(self) -> None:
        self.feats_to_ids: dict[Features, int] = {}
        self.ids_to_feats: dict[int, Features] = {}
        self.graph = nx.MultiDiGraph()
        self.ids_to_untried_acts: dict[int, Set[Action]] = {}

    def get_index(self, feats: Features, add: bool = True) -> int:
        if feats in self.feats_to_ids:
            return self.feats_to_ids[feats]
        if add:
            new_id = len(self.feats_to_ids)
            self.feats_to_ids[feats] = new_id
            self.ids_to_feats[new_id] = feats
            title = "\n".join(k for k in feats if feats[k])
            self.graph.add_node(new_id, label=f"{new_id}", title=title)
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

    def add_edge(
        self, u: Features, v: Features, a: Action, inverse_semantics: bool = True
    ) -> None:
        u_id = self.get_index(u)
        v_id = self.get_index(v)
        self._add_edge_id(u_id, v_id, a)
        self._update_untried_acts(u_id, a)
        self._remove_self_loop_on_id(u_id, a)

        if inverse_semantics and a in inverse:
            inv_a = inverse[a]
            self._add_edge_id(v_id, u_id, inv_a)
            self._update_untried_acts(v_id, inv_a)
            self._remove_self_loop_on_id(v_id, inv_a)

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
    lines = get_system(graph, with_dummy=True)
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
            "adv.txt",
            "--exportstates",
            "states.txt",
            "--exporttrans",
            "trans.txt",
        ]
    )
    reachable = parse_number_of_reachable_states(output)
    if reachable["yes"] == 0:
        output = run_bash_command(
            [
                "prism",
                "partmodel.prism",
                "--pf",
                f'Pmax=? [ ({spec}) | (F "dummy")]',
                "--exportresults",
                "stdout:comment",
                "--exportadvmdp",
                "adv.txt",
                "--exportstates",
                "states.txt",
                "--exporttrans",
                "trans.txt",
            ]
        )
    ## Check error in output here

    pmcid_id_map = parse_state_file("states.txt")
    pmcid_action_map = parse_adv_file("adv.txt")

    _ = run_bash_command(
        ["rm", "states.txt", "adv.txt", "trans.txt", "partmodel.prism"]
    )
    decisions = Decisions()
    for pmcid, act in pmcid_action_map.items():
        id = pmcid_id_map[pmcid]
        decisions.add_decision(graph.ids_to_feats[id], act)
    return decisions
