#!/usr/bin/env python3

import re
import pickle
import random
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


Specification = str

State = MiniGridEnv
Action = str
Transition = Tuple[State, Action, State]

FeaturesKey = Tuple[bool, ...]


class Features(dict):
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

    def get_decisions(self) -> List[Tuple[Features, Action]]:
        ## Conflicting decisions possible
        ## Delegates to learner to resolve conflict
        return [
            (feats.key, act)
            for feats in self.features_to_actions
            for act in self.features_to_actions[feats]
        ]
        ## Sampling a random decision from conflicts
        return [
            (feats.key, act)
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


class AbstractGraph(object):
    def __init__(self) -> None:
        self.features_to_feat_uid: dict[Features, int] = {}
        self.feat_uid_to_features: dict[int, Features] = {}
        self.feature_graph = nx.MultiDiGraph()
        self.feat_uid_to_unused_acts: dict[int, Set[Action]] = {}

    def get_index(self, feats: Features, add: bool=True) -> int:
        if feats in self.features_to_feat_uid:
            return self.features_to_feat_uid[feats]
        if add:
            new_feat_uid = len(self.features_to_feat_uid)

            self.features_to_feat_uid[feats] = new_feat_uid
            self.feat_uid_to_features[new_feat_uid] = feats
            self.feature_graph.add_node(
                new_feat_uid, label=f"{new_feat_uid}"
            )  ## Add title with features
            return new_feat_uid
        return -1

    def add_edge(self, u: Features, v: Features, a: Action) -> None:
        u_feat_uid = self.get_index(u)
        v_feat_uid = self.get_index(v)
        self.feature_graph.add_edge(u_feat_uid, v_feat_uid, key=a, label=a)
        if u_feat_uid not in self.feat_uid_to_unused_acts:
            self.feat_uid_to_unused_acts[u_feat_uid] = deepcopy(ACT_SET)
        self.feat_uid_to_unused_acts[u_feat_uid] -= {a}
        self.remove_self_loop_on(u, a)

    def remove_self_loop_on(self, u: Features, a: Action) -> None:
        u_p = self.get_index(u, add=False)
        remove = False
        for v_p in self.feature_graph[u_p]:
            if v_p != u_p and any(a == a_p for a_p in self.feature_graph[u_p][v_p]):
                remove = True
                break
        if remove and self.feature_graph.has_edge(u_p, u_p, key=a):
            self.feature_graph.remove_edge(u_p, u_p, key=a)

    def get_transitions(self, with_dummy=True) -> List[Tuple[int, int, str]]:
        transitions = [(u, v, a) for u, v, a in self.feature_graph.edges(keys=True)]
        if with_dummy:
            idx_counter = max(self.feat_uid_to_features.keys()) + 1
            for u, acts in self.feat_uid_to_unused_acts.items():
                for a in acts:
                    transitions.append((u, idx_counter, a))
                    idx_counter += 1
        return transitions

    def get_state_limits(self, with_dummy=True) -> Tuple[int, int]:
        min_idx = min(self.feat_uid_to_features.keys())
        max_idx = max(self.feat_uid_to_features.keys())
        if with_dummy:
            max_idx += sum(len(acts) for acts in self.feat_uid_to_unused_acts.values())
        return (min_idx, max_idx)

    def get_labels(self) -> dict[str, Set[int]]:
        labels = {k: set() for k in self.feat_uid_to_features[0]}
        for uid, feats in self.feat_uid_to_features.items():
            for k in labels:
                if feats[k]:
                    labels[k].add(uid)
        return labels


class PartialGraph(object):
    def __init__(self, feature_fn=None) -> None:
        self.state_id_to_state: dict[int, State] = {}
        self.state_id_to_state_uid: dict[int, int] = {}
        self.state_uid_to_state_id: dict[int, int] = {}
        self.state_graph = nx.MultiDiGraph()
        self.state_uid_to_unused_acts: dict[int, Set[Action]] = {}

        self.feature_fn = feature_fn
        self.features_to_state_ids: dict[Features, Set[int]] = {}

        self.abstract_graph = AbstractGraph()

    def get_index(self, state: State, add: bool=True) -> int:
        state_id = state.identifier()
        if state_id in self.state_id_to_state_uid:
            return self.state_id_to_state_uid[state_id]
        if add:
            new_state_uid = len(self.state_id_to_state_uid)

            self.state_id_to_state[state_id] = deepcopy(state)
            self.state_id_to_state_uid[state_id] = new_state_uid
            self.state_uid_to_state_id[new_state_uid] = state_id
            self.state_graph.add_node(
                new_state_uid, label=f"{new_state_uid}"
            )  ## title with features

            feats = self.feature_fn(state)
            if feats not in self.features_to_state_ids:
                self.features_to_state_ids[feats] = set()
            self.features_to_state_ids[feats].add(state_id)

            _ = self.abstract_graph.get_index(feats)
            return new_state_uid
        return -1

    def add_edge(self, u: State, v: State, a: Action) -> None:
        u_state_uid = self.get_index(u)
        v_state_uid = self.get_index(v)

        self.state_graph.add_edge(u_state_uid, v_state_uid, label=a, key=a)
        if u_state_uid not in self.state_uid_to_unused_acts:
            self.state_uid_to_unused_acts[u_state_uid] = deepcopy(ACT_SET)
        self.state_uid_to_unused_acts[u_state_uid] -= {a}

        self.abstract_graph.add_edge(self.feature_fn(u), self.feature_fn(v), a)

    def add_trace(self, trace: Trace) -> None:
        for u, a, v in trace:
            self.add_edge(u, v, a)

    def get_transitions(self, with_dummy=True) -> List[Tuple[int, int, str]]:
        transitions = [(u, v, a) for u, v, a in self.state_graph.edges(keys=True)]
        if with_dummy:
            idx_counter = max(self.state_uid_to_state_id.keys()) + 1
            for u, acts in self.state_uid_to_unused_acts.items():
                for a in acts:
                    transitions.append((u, idx_counter, a))
                    idx_counter += 1
        return transitions

    def get_state_limits(self, with_dummy=True) -> Tuple[int, int]:
        min_idx = min(self.state_uid_to_state_id.keys())
        max_idx = max(self.state_uid_to_state_id.keys())
        if with_dummy:
            max_idx += sum(len(acts) for acts in self.state_uid_to_unused_acts.values())
        return (min_idx, max_idx)

    def get_labels(self) -> dict[str, Set[int]]:
        f_ex = self.feature_fn(self.state_id_to_state[self.state_uid_to_state_id[0]])
        labels = {k: set() for k in f_ex}
        for feats in self.features_to_state_ids:
            for k in labels:
                if feats[k]:
                    for state_id in self.features_to_state_ids[feats]:
                        labels[k].add(self.state_id_to_state_uid[state_id])
        return labels


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
        default='F "is_agent_on__goal"',
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
    parsed_args = parser.parse_args()

    args = Arguments(
        env_name=parsed_args.env_name,
        spec=parsed_args.spec,
        simulator_seed=parsed_args.simulator_seed,
        demo_seed=parsed_args.demo_seed,
        max_steps=parsed_args.max_steps,
        tile_size=parsed_args.tile_size,
        threshold=parsed_args.threshold,
    )
    return args


def run_bash_command(bash_command: List[str]) -> str:
    """Runs a bash command given as a list of strings"""
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    return output.decode("utf-8")


def get_specification_result(output: str) -> bool:
    """Check if the PRISM output for a given specification check returned True"""
    result_str = "// RESULT: "
    index = output.find(result_str)
    if index == -1:
        raise Exception("Could not compute the spec the result")
    st_index = index + len(result_str)
    return output[st_index : st_index + 4] == "true"


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


state_exp = re.compile(r"(?P<pmcid>[-+]?\d+)\:\((?P<idx>[-+]?\d+)\)")


def parse_state_file(state_file: str) -> dict[int, int]:
    """Parse state file"""
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
    """Parse adversary file"""
    pmcid_action_map = {}
    with open(adv_file, "r") as f:
        _ = f.readline()
        for line in f:
            d = transition_exp.match(line)
            pmcid_action_map[int(d["upmcid"])] = Action(d["act"])
    return pmcid_action_map


def get_system(mdp: AbstractGraph | PartialGraph, with_dummy=True, only_start_init=False) -> List[str]:
    state_min, dummy_max = mdp.get_state_limits(with_dummy=with_dummy)
    if dummy_max == state_min:
        dummy_max += 1
    state_lines = ["// World State", f"state: [{state_min}..{dummy_max}];", ""]

    transitions_lines = [
        f"[{a}] (state={u}) -> (state'={v}) ;"
        for u, v, a in mdp.get_transitions(with_dummy=with_dummy)
    ] + [""]

    if only_start_init:
        init_lines = [f"init state={state_min} endinit", ""]
    else:
        _, state_max = mdp.get_state_limits(with_dummy=False)
        init_lines = [f"init state<={state_max} endinit", ""]

    labels = mdp.get_labels()
    label_lines = []
    for l in labels:
        sat_states = " | ".join(f"(state={s})" for s in labels[l])
        if sat_states == "":
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


def satisfies(trace: Trace, spec: Specification, feature_fn: Feature_Func) -> bool:
    partialmdp = PartialGraph(feature_fn)
    partialmdp.add_trace(trace)
    lines = get_system(partialmdp, with_dummy=False, only_start_init=True)
    with open("model.prism", "w") as f:
        f.writelines(lines)
    output = run_bash_command(
        [
            "prism",
            "model.prism",
            "--pf",
            f"P>=1 [ {spec} ]",
            "--exportresults",
            "stdout:comment",
        ]
    )
    _ = run_bash_command(["rm", 'model.prism'])
    return get_specification_result(output=output)


def get_decisions(mdp: PartialGraph, spec: Specification) -> Decisions:
    lines = get_system(mdp.abstract_graph, with_dummy=True, only_start_init=False)
    with open("partmodel.prism", "w") as f:
        f.writelines(lines)
    output = run_bash_command(
        [
            "prism",
            "partmodel.prism",
            "--pf",
            f'Pmax=? [ ({spec}) | (F "dummy") ]',
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
    nt = Network("500px", "500px", directed=True)
    nt.from_nx(mdp.abstract_graph.feature_graph)
    nt.show_buttons(filter_=["physics"])
    nt.show("graph.html")
    # debug(output)
    ## Check error in output here

    pmcid_uid_map = parse_state_file("states.txt")
    pmcid_action_map = parse_adv_file("adv.txt")

    output = run_bash_command(["rm", 'states.txt', 'adv.txt', 'trans.txt', 'partmodel.prism'])
    decisions = Decisions()
    for pmcid, act in pmcid_action_map.items():
        uid = pmcid_uid_map[pmcid]
        decisions.add_decision(mdp.abstract_graph.feat_uid_to_features[uid], act)
    return decisions
