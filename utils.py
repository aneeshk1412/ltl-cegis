#!/usr/bin/env python3

import pickle
import pandas as pd
from typing import Tuple

from dsl_minigrid import header_register, feature_register
from minigrid.minigrid_env import MiniGridEnv

DEBUG = True


def debug(*args, **kwargs):
    if DEBUG or kwargs["debug"]:
        print(*args, **kwargs)


def get_augmented_policy(policy, speculation_set, env_name):
    def augmented_policy(env):
        state = feature_register[env_name](env)
        if state in speculation_set:
            return speculation_set[state]
        else:
            return policy(env)

    return augmented_policy


def env_to_state(env: MiniGridEnv, env_name: str) -> Tuple[bool, ...]:
    return feature_register[env_name](env)


def state_to_bitstring(state: Tuple[bool, ...]) -> str:
    return "".join(str(int(s)) for s in state)


def bitstring_to_state(s: str) -> Tuple[bool, ...]:
    return tuple(c == "1" for c in s)


def state_to_string(state: Tuple[bool, ...], env_name: str) -> str:
    return "\n".join(header_register[env_name][i] for i, s in enumerate(state) if s)


def state_to_pretty_string(state: Tuple[bool, ...], env_name: str) -> str:
    return "    ".join(
        intersperse(
            items=[header_register[env_name][i] for i in range(len(state)) if state[i]],
            sep="\n",
            space=2,
        )
    )


def bitstring_to_string(s: str, env_name: str) -> str:
    return "\n".join(header_register[env_name][i] for i, c in enumerate(s) if c == "1")


def intersperse(items, sep, space=3):
    return [
        x
        for y in (
            items[i : i + space] + [sep] * (i < len(items) - (space - 1))
            for i in range(0, len(items), space)
        )
        for x in y
    ]


def load_envs_from_pickle(env_name: str):
    env_list = []
    try:
        with open("data/" + env_name + "-envs.pkl", "rb") as f:
            while True:
                try:
                    env_list.append(pickle.load(f))
                except EOFError:
                    break
    except FileNotFoundError:
        pass
    return env_list


def demo_traces_to_pickle(demos, env_name: str):
    with open("data/" + env_name + "-demos.pkl", "wb") as f:
        pickle.dump(demos, f)


def pickle_to_demo_traces(env_name: str):
    with open("data/" + env_name + "-demos.pkl", "rb") as f:
        positive_demos = pickle.load(f)
    return positive_demos


def demos_to_positive_samples_csv(demos, env_name: str):
    d = dict()
    for trace in demos:
        for _, s, a, _, _ in trace:
            assert s not in d or d[s] == a
            d[s] = a
    df = pd.DataFrame(
        list(set(tuple(list(s) + [a]) for trace in demos for _, s, a, _, _ in trace)),
        columns=header_register[env_name] + tuple(["action"]),
    )
    df.to_csv("data/" + env_name + "-demos.csv", index=False)


def csv_to_positive_samples_dict(env_name: str):
    df = pd.read_csv("data/" + env_name + "-demos.csv")
    records = df.to_dict("records")
    records = [
        (tuple(record[key] for key in header_register[env_name]), record["action"])
        for record in records
    ]
    return dict(records)
