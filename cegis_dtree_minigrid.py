#!/usr/bin/env python3

import pickle
import pandas as pd
from sklearn import tree
from pprint import pprint
import matplotlib.pyplot as plt
import random

from minigrid.core.constants import ACT_KEY_TO_IDX

from verifier_minigrid import verify_action_selection_policy
from demos_gen_minigrid import generate_demonstrations


def get_stem_and_loop(trace):
    hashes = [str(env) for env, _ in trace]
    for i, x in enumerate(hashes):
        try:
            idx = hashes[i + 1 :].index(x) + i + 1
            return trace[:i], trace[i:idx]
        except ValueError:
            continue
    return trace, None


def train_and_save_model(positive_dict, negative_dict, extra_dict=None, seed=None):
    if extra_dict is None:
        assert set(positive_dict.keys()) & set(negative_dict.keys()) == set([])
        state_demos = pd.DataFrame(
            [state for state in positive_dict] + [state for state in negative_dict]
        )
        act_demos = pd.DataFrame(
            [positive_dict[state] for state in positive_dict]
            + [negative_dict[state] for state in negative_dict]
        )
    else:
        assert set(positive_dict.keys()) & set(negative_dict.keys()) == set([])
        state_demos = pd.DataFrame(
            [state for state in positive_dict]
            + [state for state in negative_dict]
            + [state for state in extra_dict]
        )
        act_demos = pd.DataFrame(
            [positive_dict[state] for state in positive_dict]
            + [negative_dict[state] for state in negative_dict]
            + [extra_dict[state] for state in extra_dict]
        )
    aspmodel = tree.DecisionTreeClassifier(
        class_weight="balanced",
        random_state=seed,
        max_features=None,
        max_leaf_nodes=None,
    )
    clf = aspmodel.fit(state_demos, act_demos)
    with open("DT.model", "wb") as f:
        pickle.dump(clf, f)
    return clf


def print_trace_stem_loop(trace, stem, loop):
    for line in trace:
        pprint(line)
    print()
    print("STEM: ")
    for line in stem:
        pprint(line)
    print()
    print("LOOP: ")
    if loop is not None:
        for line in loop:
            pprint(line)
        print()


def random_sampling_algorithm(positive_dict, trace):
    for _, obs, act in reversed(trace):
        if obs in positive_dict:
            continue
        new_act = random.sample([a for a in ACT_KEY_TO_IDX.keys() if a != act], 1)
        return obs, new_act[0]
    return None, None


def random_loop_correction(positive_dict, trace, demo_envs):
    stem, loop = get_stem_and_loop(trace, demo_envs)
    if loop is None:
        return random_sampling_algorithm(positive_dict, stem)
    for x, y in loop:
        if x in positive_dict:
            continue
        z = random.sample(list(set(ACT_KEY_TO_IDX.keys()) - set([y])), 1)
        return x, z[0]
    return None, None


if __name__ == "__main__":
    import argparse
    from asp_minigrid import ground_truth_asp_register, action_selection_policy_decision_tree
    from dsl_minigrid import feature_register, header_register

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        help="gym environment to load",
        default="MiniGrid-DoorKey-16x16-v0",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="timeout to complete the task",
        default=100,
    )


    parser.add_argument(
        "--demo-seed",
        type=int,
        help="random seed to generate demonstrations",
        default=None,
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        help="number of demonstrations to run",
        default=1,
    )
    parser.add_argument(
        "--select-partial-demos",
        default=False,
        help="whether to use complete demonstration or select a substring",
        action="store_true",
    )


    parser.add_argument(
        "--verifier-seed",
        type=int,
        help="random seed for the model checker",
        default=None,
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        help="number of trials to verify on",
        default=100,
    )


    parser.add_argument(
        "--tree-seed",
        type=int,
        help="random seed for the tree learning algorithm",
        default=None,
    )
    parser.add_argument(
        "--plot-tree",
        default=False,
        help="whether to show the decision tree",
        action="store_true",
    )


    parser.add_argument(
        "--show-window",
        default=False,
        help="whether to show the animation window",
        action="store_true",
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    positive_demos = generate_demonstrations(
        args.env_name,
        ground_truth_asp_register[args.env_name],
        feature_register[args.env_name],
        seed=args.demo_seed,
        num_demos=args.num_demos,
        timeout=args.timeout,
        select_partial_demos=args.select_partial_demos,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    positive_dict = dict((obs, act) for _, obs, act in positive_demos)
    negative_dict = dict()

    sat = False
    epoch = 0
    while not sat:
        print(f"{epoch = }")
        aspmodel = train_and_save_model(positive_dict, negative_dict, seed=args.tree_seed)

        action_selection_policy = lambda env: action_selection_policy_decision_tree(env, aspmodel, feature_register[args.env_name])
        sat, trace = verify_action_selection_policy(
            args.env_name,
            action_selection_policy,
            feature_register[args.env_name],
            seed=args.verifier_seed,
            num_trials=args.num_trials,
            timeout=args.timeout,
            show_window=args.show_window,
            tile_size=args.tile_size,
            agent_view=args.agent_view,
        )

        print(f"{sat = }")
        if not sat:
            x, z = random_sampling_algorithm(positive_dict, trace)
            negative_dict[x] = z
            print(f"Added Demonstration: {x} -> {z}")
        print()
        epoch += 1

    if args.plot_tree:
        tree.plot_tree(
            aspmodel,
            max_depth=None,
            class_names=sorted(set(positive_dict.values()) | set(negative_dict.values())),
            label="none",
            precision=1,
            feature_names=header_register[args.env_name],
            rounded=True,
            fontsize=5,
            proportion=True,
        )
        plt.show()
