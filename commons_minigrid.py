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
        self.key: FeaturesKey = tuple(
            self[k] for k in sorted(self.keys())
        )

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


class PartialMDP(object):
    def __init__(self) -> None:
        self.ids_to_idxs: dict[int, int] = {}
        self.idxs_to_ids: dict[int, int] = {}
        self.idxs_to_states: dict[int, State] = {}
        self.idxs_to_unused_acts: dict[int, Set[Action]] = {}
        self.graph = nx.MultiDiGraph()

    def contains(self, state_id: int) -> bool:
        return state_id in self.ids_to_idxs

    def get_index_of(self, state: State, add=True) -> int:
        state_id = state.identifier()
        if self.contains(state_id):
            return self.ids_to_idxs[state_id]
        if add:
            new_idx = len(self.ids_to_idxs)
            self.ids_to_idxs[state_id] = new_idx
            self.idxs_to_ids[new_idx] = state_id
            self.idxs_to_states[new_idx] = deepcopy(state)
            return new_idx
        return -1

    def add_trace(self, trace: Trace) -> None:
        self.graph.add_edges_from(
            [
                (self.get_index_of(s), self.get_index_of(s_p), {"act": a, "label": a})
                for s, a, s_p in trace
            ]
        )
        for s, a, _ in trace:
            idx = self.get_index_of(s)
            if idx not in self.idxs_to_unused_acts:
                self.idxs_to_unused_acts[idx] = deepcopy(ACT_SET)
            self.idxs_to_unused_acts[idx] -= {a}

    def get_mdp_line(self) -> List[str]:
        return ["mdp", ""]

    def get_module_lines(self) -> List[str]:
        return (
            ["module System", ""]
            + self.get_state_lines()
            + self.get_transition_lines()
            + ["endmodule", ""]
        )

    def get_module_with_dummy_lines(self) -> List[str]:
        return (
            ["module System", ""]
            + self.get_state_with_dummy_lines()
            + self.get_transition_lines()
            + self.get_dummy_lines()
            + ["endmodule", ""]
        )

    def get_dummy_lines(self) -> List[str]:
        idx_counter = max(self.idxs_to_ids.keys()) + 1
        transitions = []
        for u, acts in self.idxs_to_unused_acts.items():
            for a in acts:
                transitions.append(f"[{a}] (state={u}) -> (state'={idx_counter}) ;")
                idx_counter += 1
        return transitions

    def get_state_with_dummy_lines(self) -> List[str]:
        min_idx = min(self.idxs_to_ids.keys())
        max_idx = max(self.idxs_to_ids.keys()) + sum(
            len(v) for v in self.idxs_to_unused_acts.values()
        )
        return ["// World State", f"state: [{min_idx}..{max_idx}] ;", ""]

    def get_state_lines(self) -> List[str]:
        min_idx = min(self.idxs_to_ids.keys())
        max_idx = max(self.idxs_to_ids.keys())
        return ["// World State", f"state: [{min_idx}..{max_idx}] ;", ""]

    def get_transition_lines(self):
        transitions = [
            f"[{a}] (state={u}) -> (state'={v}) ;"
            for u, v, a in self.graph.edges(data="act")
        ]
        return ["// Transitions"] + transitions + [""]

    def get_init_state_line(self, init_states: List[State] | None = None):
        if init_states is None or len(init_states) == 0:
            ## All states are init (except dummy states)
            init_states_cond = " | ".join(
                f"(state={idx})" for idx in self.idxs_to_states
            )
        else:
            init_states_cond = " | ".join(
                f"(state={self.ids_to_idxs[s.identifier()]})" for s in init_states
            )
        return [f"init {init_states_cond} endinit"]

    def get_label_lines(self, feature_fn: Feature_Func):
        labels = {k: set() for k in feature_fn(self.idxs_to_states[0])}
        for s_idx in self.idxs_to_states:
            for k, v in feature_fn(self.idxs_to_states[s_idx]).items():
                if v:
                    labels[k].add(s_idx)
        lines = []
        for l in labels:
            sat_states = " | ".join(f"(state={s})" for s in labels[l])
            if sat_states == "":
                sat_states = "false"
            feat = f'label "{l}" = ' + sat_states + ";"
            lines.append(feat)
        return lines + [""]

    def get_dummy_label_line(self):
        return [f'label "dummy" = state>{max(self.idxs_to_ids.keys())} ;', ""]

    def get_partial_mdp_lines(self, feature_fn: Feature_Func) -> List[str]:
        lines = (
            self.get_mdp_line()
            + self.get_module_lines()
            + self.get_init_state_line()
            + self.get_label_lines(feature_fn=feature_fn)
        )
        return [l + "\n" for l in lines]

    def get_mdp_with_dummy_lines(self, feature_fn: Feature_Func) -> List[str]:
        lines = (
            self.get_mdp_line()
            + self.get_module_with_dummy_lines()
            + self.get_init_state_line()
            + self.get_label_lines(feature_fn=feature_fn)
            + self.get_dummy_label_line()
        )
        return [l + "\n" for l in lines]

    def get_decisions(self, spec: Specification, feature_fn: Feature_Func) -> Decisions:
        lines = self.get_mdp_with_dummy_lines(feature_fn=feature_fn)
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
        nt.from_nx(self.graph)
        nt.show_buttons(filter_=['physics'])
        nt.show("graph.html")
        # debug(output)
        ## Check error in output here

        pmcid_idx_map = parse_state_file("states.txt")
        pmcid_action_map = parse_adv_file("adv.txt")

        decisions = Decisions()
        for pmcid, act in pmcid_action_map.items():
            idx = pmcid_idx_map[pmcid]
            decisions.add_decision(feature_fn(self.idxs_to_states[idx]), act)
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
    """ Runs a bash command given as a list of strings"""
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    return output.decode("utf-8")


def get_specification_result(output: str) -> bool:
    """ Check if the PRISM output for a given specification check returned True """
    result_str = "// RESULT: "
    index = output.find(result_str)
    if index == -1:
        raise Exception("Could not compute the spec the result")
    st_index = index + len(result_str)
    return output[st_index : st_index + 4] == "true"


def get_stem_and_loop(
    trace: List[Transition],
) -> Tuple[List[Transition], List[Transition] | None]:
    """ Get stem and loop for a given trace """
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
    """ Save the list of Demo traces to a pickle file """
    with open("data/" + env_name + "-demos.pkl", "wb") as f:
        pickle.dump(demos, f)


def pickle_to_demo_traces(env_name: str) -> List[Trace]:
    """ Load a list of Demo traces from a pickle file """
    with open("data/" + env_name + "-demos.pkl", "rb") as f:
        positive_demos = pickle.load(f)
    return positive_demos


state_exp = re.compile(r"(?P<pmcid>[-+]?\d+)\:\((?P<idx>[-+]?\d+)\)")


def parse_state_file(state_file: str) -> dict[int, int]:
    """ Parse state file """
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
    """ Parse adversary file """
    pmcid_action_map = {}
    with open(adv_file, "r") as f:
        _ = f.readline()
        for line in f:
            d = transition_exp.match(line)
            pmcid_action_map[int(d["upmcid"])] = Action(d["act"])
    return pmcid_action_map


def satisfies(trace: Trace, spec: Specification, feature_fn: Feature_Func) -> bool:
    partialmdp = PartialMDP()
    partialmdp.add_trace(trace)
    lines = partialmdp.get_partial_mdp_lines(feature_fn=feature_fn)
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
    return get_specification_result(output=output)
