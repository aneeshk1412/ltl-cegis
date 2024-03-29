#!/usr/bin/env python3

# cmd to run:
# python3 cegis_dtree_minigrid.py --env-name MiniGrid-DoorKey-16x16-v0 --num-demos 2 --num-trials 100 --show-window --plot-tree


import pickle
import pandas as pd
from sklearn import tree
from pprint import pprint
import matplotlib.pyplot as plt
import random
from copy import deepcopy

from minigrid.core.constants import ACT_KEY_TO_IDX

from verifier_minigrid import (
    verify_action_selection_policy,
    verify_action_selection_policy_on_env,
)
from demos_gen_minigrid import generate_demonstrations


def get_stem_and_loop(trace):
    hashes = [str(env) for env, _, _ in trace]
    for i, x in enumerate(hashes):
        try:
            idx = hashes[i + 1 :].index(x) + i + 1
            return trace[:i], trace[i:idx]
        except ValueError:
            continue
    return trace, None


def train_and_save_model(
    demonstrations_sample_dict, speculated_sample_dict, extra_dict=None, seed=None
):
    if extra_dict is None:
        assert set(demonstrations_sample_dict.keys()) & set(
            speculated_sample_dict.keys()
        ) == set([])
        state_demos = pd.DataFrame(
            [state for state in demonstrations_sample_dict]
            + [state for state in speculated_sample_dict]
        )
        act_demos = pd.DataFrame(
            [demonstrations_sample_dict[state] for state in demonstrations_sample_dict]
            + [speculated_sample_dict[state] for state in speculated_sample_dict]
        )
    else:
        assert set(demonstrations_sample_dict.keys()) & set(
            speculated_sample_dict.keys()
        ) == set([])
        state_demos = pd.DataFrame(
            [state for state in demonstrations_sample_dict]
            + [state for state in speculated_sample_dict]
            + [state for state in extra_dict]
        )
        act_demos = pd.DataFrame(
            [demonstrations_sample_dict[state] for state in demonstrations_sample_dict]
            + [speculated_sample_dict[state] for state in speculated_sample_dict]
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


def get_unique_states_from_trace(trace):
    return list(set((obs, act) for _, obs, act in trace))


def random_sampling_algorithm(demonstrations_sample_dict, trace):
    for obs, act in reversed(get_unique_states_from_trace(trace)):
        if obs in demonstrations_sample_dict:
            continue
        new_act = random.sample([a for a in ACT_KEY_TO_IDX.keys() if a != act], 1)
        return obs, new_act[0]
    return None, None


def random_loop_correction(demonstrations_sample_dict, trace):
    stem, loop = get_stem_and_loop(trace)
    if loop is None:
        return random_sampling_algorithm(demonstrations_sample_dict, stem)
    for obs, act in get_unique_states_from_trace(loop):
        if obs in demonstrations_sample_dict:
            continue
        new_act = random.sample([a for a in ACT_KEY_TO_IDX.keys() if a != act], 1)
        return obs, new_act[0]
    return None, None


def bounded_dfs_single_trace(args, demonstrations_sample_dict, speculated_sample_dict, trace):
    current_speculated_sample_dict = deepcopy(speculated_sample_dict)
    first_env, _, _ = trace[0]
    new_speculated_sample_dict = bounded_dfs_single_trace_helper(
        args,
        demonstrations_sample_dict,
        current_speculated_sample_dict,
        trace,
        first_env,
    )
    return new_speculated_sample_dict


def bounded_dfs_single_trace_helper(
    args,
    demonstrations_sample_dict,
    speculated_sample_dict,
    trace,
    first_env,
    bound=3,
):
    if bound == 0:
        return dict()
    for obs, act in reversed(get_unique_states_from_trace(trace)):
        if obs in demonstrations_sample_dict:
            continue
        ## Change the label of the last observation which is not in demonstration samples
        for new_act in ACT_KEY_TO_IDX.keys():
            if new_act != act:
                print(f"{(bound, obs, new_act) =}")
                # print(f"{get_unique_states_from_trace(trace)}")
                speculated_sample_dict[obs] = new_act
                aspmodel = train_and_save_model(
                    demonstrations_sample_dict,
                    speculated_sample_dict,
                    seed=args.tree_seed,
                )
                action_selection_policy = (
                    lambda env: action_selection_policy_decision_tree(
                        env, aspmodel, feature_register[args.env_name]
                    )
                )

                sat, new_trace = verify_action_selection_policy_on_env(
                    first_env,
                    action_selection_policy,
                    feature_register[args.env_name],
                    seed=args.verifier_seed,
                    timeout=args.timeout,
                    show_window=args.show_window,
                    tile_size=args.tile_size,
                    agent_view=args.agent_view,
                )
                if sat:
                    ## Found a satisfying speculation for this first_env
                    return deepcopy(speculated_sample_dict)
                else:
                    ## Recursion
                    # print(f"{get_unique_states_from_trace(new_trace)}")
                    new_speculated_sample_dict = bounded_dfs_single_trace_helper(
                        args,
                        demonstrations_sample_dict,
                        speculated_sample_dict,
                        new_trace,
                        first_env,
                        bound - 1,
                    )
                    if new_speculated_sample_dict:
                        return new_speculated_sample_dict


if __name__ == "__main__":
    import argparse
    from asp_minigrid import (
        ground_truth_asp_register,
        action_selection_policy_decision_tree,
    )
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
        help="draw what the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    num_demos = 1 if args.num_demos == 0 else args.num_demos
    positive_demonstrations_list = generate_demonstrations(
        args.env_name,
        ground_truth_asp_register[args.env_name],
        feature_register[args.env_name],
        seed=args.demo_seed,
        num_demos=num_demos,
        timeout=args.timeout,
        select_partial_demos=args.select_partial_demos,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
    )
    if args.num_demos == 0:
        positive_demonstrations_list = positive_demonstrations_list[:1]
    demonstrations_sample_dict = dict(
        (obs, act) for _, obs, act in positive_demonstrations_list
    )
    speculated_sample_dict = dict()

    sat = False
    epoch = 0
    while not sat:
        print(f"{epoch = }")
        aspmodel = train_and_save_model(
            demonstrations_sample_dict, speculated_sample_dict, seed=args.tree_seed
        )

        action_selection_policy = lambda env: action_selection_policy_decision_tree(
            env, aspmodel, feature_register[args.env_name]
        )
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
            epoch=epoch,
            use_known_error_envs=True,
        )

        print(f"{sat = }")
        if not sat:
            new_speculated_sample_dict = bounded_dfs_single_trace(
                args, demonstrations_sample_dict, speculated_sample_dict, trace
            )
            if new_speculated_sample_dict:
                print("")
                print(f"Found a new speculation:")
                pprint(new_speculated_sample_dict)
                for obs, act in new_speculated_sample_dict.items():
                    ## This might rewrite stuff though
                    if obs in speculated_sample_dict:
                        print("Rewrite occured...")
                    speculated_sample_dict[obs] = act
            else:
                print(f"Fallback to Random Sampling...")
                obs, act = random_sampling_algorithm(demonstrations_sample_dict, trace)
                speculated_sample_dict[obs] = act
                print(f"Added Demonstration: {obs} -> {act}")

        print()
        epoch += 1

    if args.plot_tree:
        tree.plot_tree(
            aspmodel,
            max_depth=None,
            class_names=sorted(
                set(demonstrations_sample_dict.values())
                | set(speculated_sample_dict.values())
            ),
            label="none",
            precision=1,
            feature_names=header_register[args.env_name],
            rounded=True,
            fontsize=5,
            proportion=True,
        )
        plt.show()
