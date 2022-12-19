#!/usr/bin/env python3

# cmd to run:
# python3 cegis_dtree_minigrid.py --env-name MiniGrid-DoorKey-16x16-v0 --num-demos 2 --num-trials 100 --show-window --plot-tree




### notes:
# I ran some experiments. I first attempted to keep a history of labels assigned to an obs and never repeat labels for the same obs. However this
# seemed to perform poorly because sometimes we need to reconsider a previously considered label again (i.e. we need to back track on decisions some times)
# I also changes the learning algorithm to stick to the same env (once a CEx is detected there) until that env is solved


import pickle
import pandas as pd
from sklearn import tree
from pprint import pprint
import matplotlib.pyplot as plt
import random

from minigrid.core.constants import ACT_KEY_TO_IDX

from verifier_minigrid import verify_action_selection_policy, verify_action_selection_policy_on_env
from demos_gen_minigrid import generate_demonstrations
from utils import bool_to_bit_vec



def get_stem_and_loop(trace):
    hashes = [str(env) for env, _, _ in trace]
    for i, x in enumerate(hashes):
        try:
            idx = hashes[i + 1 :].index(x) + i + 1
            return trace[:i], trace[i:idx]
        except ValueError:
            continue
    return trace, None

# negative dict is not a good variable name. These are not 'negative' samples. They are samples which are mutable
def train_and_save_model(positive_dict, mutable_dict, extra_dict=None, seed=None):
    if extra_dict is None:
        assert set(positive_dict.keys()) & set(mutable_dict.keys()) == set([]) # positive and negative samples are disjoint
        state_demos = pd.DataFrame(
            [state for state in positive_dict] + [state for state in mutable_dict]
        )
        act_demos = pd.DataFrame(
            [positive_dict[state] for state in positive_dict]
            + [mutable_dict[state][-1] for state in mutable_dict]
        )
    else:
        raise Exception('i do not yet know what this branch is doing')
        assert set(positive_dict.keys()) & set(mutable_dict.keys()) == set([])
        state_demos = pd.DataFrame(
            [state for state in positive_dict]
            + [state for state in mutable_dict]
            + [state for state in extra_dict]
        )
        act_demos = pd.DataFrame(
            [positive_dict[state] for state in positive_dict]
            + [mutable_dict[state] for state in mutable_dict]
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


def random_sampling_algorithm(positive_dict, mutable_dict, trace):
    for _, obs, act in reversed(trace):
        if obs in positive_dict:
            continue
        # here I tried to keep a history of labels assigned to each obs and never try them again. But does not seem to work well
        existing_labs = mutable_dict[obs] if obs in mutable_dict else []
        candidate_labs = [a for a in ACT_KEY_TO_IDX.keys() if a != act and (not a in existing_labs) and (a not in ['enter', 'pagedown'])]
        if candidate_labs == []:
            continue

        # original policy (where labels can always be overwritten)
        # candidate_labs = [a for a in ACT_KEY_TO_IDX.keys() if a != act and a not in ['enter', 'pagedown']]

        new_act = random.sample(candidate_labs, 1)
        return obs, new_act[0]
    return None, None


#def random_loop_correction(positive_dict, trace):
#    stem, loop = get_stem_and_loop(trace)
#    if loop is None:
#        return random_sampling_algorithm(positive_dict, stem)
#    for _, obs, act in loop:
#        if obs in positive_dict:
#            continue
#        new_act = random.sample([a for a in ACT_KEY_TO_IDX.keys() if a != act], 1)
#        return obs, new_act[0]
#    return None, None


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
    positive_dict = dict((obs, act) for _, obs, act in positive_demos) # 
    # modification: mutable_dict will contain the history of all labels assigned to a state. The last element in the list represents the currently considered label
    mutable_dict = dict()


    sat = False
    epoch = 0

    last_error_env = None
    stick_to_one_env_flag = False


    while not sat:
        print(f"{epoch = }")
        aspmodel = train_and_save_model(positive_dict, mutable_dict, seed=args.tree_seed)
        action_selection_policy = lambda env: action_selection_policy_decision_tree(env, aspmodel, feature_register[args.env_name])
        
        # verify the learned model 
        if not stick_to_one_env_flag:
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
                epoch=epoch
            )
        else:
            sat, trace = verify_action_selection_policy_on_env(
                env=last_error_env,
                action_selection_policy=action_selection_policy,
                observation_function=feature_register[args.env_name],
                seed=args.verifier_seed,
                timeout=args.timeout,
                show_window=args.show_window,
                tile_size=args.tile_size,
                agent_view=args.agent_view,
                epoch=epoch
            )

        print(f"{sat = }")
        if not sat:
            stick_to_one_env_flag = True
            last_error_env = trace [0][0]

            obs, act = random_sampling_algorithm(positive_dict, mutable_dict, trace)
            if obs in mutable_dict:
                mutable_dict[obs].append(act)
            else:
                mutable_dict[obs] = [act]
            print(f"Added Demonstration: {obs} -> {act}")
        else:
            if stick_to_one_env_flag:
                sat = False # try again with the random verifier
                stick_to_one_env_flag = False
                last_error_env = None
        print()
        epoch += 1



    if args.plot_tree:
        tree.plot_tree(
            aspmodel,
            max_depth=None,
            class_names=sorted(set(positive_dict.values()) | set(mutable_dict.values())),
            label="none",
            precision=1,
            feature_names=header_register[args.env_name],
            rounded=True,
            fontsize=5,
            proportion=True,
        )
        plt.show()
