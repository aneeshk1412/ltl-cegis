#!/usr/bin/env python3

# cmd to run:
# python3 cegis_dtree_minigrid.py --env-name MiniGrid-DoorKey-16x16-v0 --num-demos 2 --num-trials 100 --show-window --plot-tree


import csv
import pickle
import pandas as pd
from sklearn import tree
from pprint import pprint
import matplotlib.pyplot as plt
import random

from minigrid.core.constants import ACT_KEY_TO_IDX
from tabulate import tabulate
from dsl_minigrid import env_state_to_readable_str, readable_headers_list

from verifier_minigrid import load_all_pickle, verify_action_selection_policy
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


def random_loop_correction(positive_dict, trace):
    stem, loop = get_stem_and_loop(trace)
    if loop is None:
        return random_sampling_algorithm(positive_dict, stem)
    for _, obs, act in loop:
        if obs in positive_dict:
            continue
        new_act = random.sample([a for a in ACT_KEY_TO_IDX.keys() if a != act], 1)
        return obs, new_act[0]
    return None, None



def one_shot_learning(args):
    # init variabales
    show_tree = False

    # run demonstrations from the ground truth and save them to samples.csv file -- need not be called all the time
    # generate_and_save_samples(args)

    # load existing samples from samples.csv file (this file initially contains samples from 2 correct demonstrations)
    samples = []
    state_demos = []
    act_demos = []
    with open('samples.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for line in csv_reader:
            if '#' in line[0]:
                continue
            samples.append(line)
            act_demos.append(line[-1])
            state_demos.append([eval(x) for x in line[:-1]])


    # train a BDT model based on the samples
    aspmodel = tree.DecisionTreeClassifier(class_weight="balanced", random_state=args.tree_seed)
    bdt_model = aspmodel.fit(state_demos, act_demos)
    if show_tree:
        tree.plot_tree(
                bdt_model,
                max_depth=None,
                class_names=sorted(set(act_demos)),
                label="none",
                precision=1,
                feature_names=header_register[args.env_name],
                rounded=True,
                fontsize=5,
                proportion=True,
            )
        plt.show()

    # verify the model and generate a number of counter-examples (verification consists of 100 random tests)
    action_selection_policy = lambda env: action_selection_policy_decision_tree(env, bdt_model, feature_register[args.env_name])
    sats, traces = verify_action_selection_policy(
            args.env_name,
            action_selection_policy,
            feature_register[args.env_name],
            seed=args.verifier_seed,
            num_trials=args.num_trials,
            timeout=args.timeout,
            show_window=args.show_window,
            tile_size=args.tile_size,
            agent_view=args.agent_view,
            use_known_error_envs = False,
            verify_each_step_manually = False, 
            cex_count = 3,
        )
    print(f"{sats = }")

    # suggest a (set of) new samples based on the counter-examples (try to automate the insights from above step)
    repair = analyze_traces_suggest_repair(traces) 
    print (repair)
    

def analyze_traces_suggest_repair(traces):
    manually_added_samples = [[False,False,False,False,False,True,False,False,True,False,True,True,False,False,False,False,False], \
        [False,False,False,False,False,True,False,True,False,False,True,False,False,True,False,False,False],
        [False,False,False,False,False,True,True,True,False,False,True,False,True,False,False,False,False],
        [False,False,False,False,False,True,True,False,True,False,True,False,False,False,False,False,False],
        [False,False,False,False,False,True,True,False,True,False,True,False,False,True,False,False,False],
        [False,False,False,False,False,True,True,False,True,False,False,False,False,False,False,False,True],
        [False,False,False,False,False,True,False,False,True,False,True,False,True,False,False,False,True],
        [False,False,False,False,False,True,True,False,False,True,True,False,False,True,False,False,False],
        [False,False,False,False,False,True,True,False,False,True,True,False,True,False,False,False,False]]
    # a dictionary to keep track of number of occurences of the manually added samples with repetition per trace
    mas_cnt_with_rep = {tuple(x):0 for x in manually_added_samples}
    mas_cnt_without_rep = {tuple(x):0 for x in manually_added_samples}
    total_trace_lens = 0
    trace_cnt = 0

    
    event_map = dict() # number of each state occurence in the given traces
    action_map = dict()
    for trace in traces:
        trace_cnt += 1
        seen_bvs = set()
        for env, bv, a in trace:
            total_trace_lens += 1
            if bv in mas_cnt_with_rep:
                mas_cnt_with_rep[bv] = mas_cnt_with_rep[bv] + 1
            if bv in event_map:
                event_map[bv] = event_map[bv] + 1
                action_map[bv].add(a)
            else:
                event_map[bv] = 1
                action_map[bv] = set([a])
            # check for repetitions
            if bv in seen_bvs:
                continue
            else:
                seen_bvs.add(bv)
            # increment counter without reps
            if bv in mas_cnt_without_rep:
                mas_cnt_without_rep[bv] = mas_cnt_without_rep[bv] + 1

    i = 0 
    res = [['feature']  + ['action taken', 'number of occurences'] + readable_headers_list() ]
    for s in event_map:
        res.append(['event#'+str(i)] + list(action_map[s]) + [event_map[s]] + [x for x in s] )
        i += 1
    transposed_data = list(zip(*res))
    print ('='*100)
    i = 0
    for bv in manually_added_samples:
        print ('bv#'+str(i), str(mas_cnt_with_rep[tuple(bv)]).ljust(4,' '), 
        str(mas_cnt_without_rep[tuple(bv)]).ljust(4,' '), bv)
        i += 1
    print ('total trace lens:', total_trace_lens)
    print ('trace count:', trace_cnt)
    res = ''
    for s in event_map:
        res += (str(event_map[s]) + ', ')
    print (res)
    #print (tabulate(transposed_data))


def generate_and_save_samples(args):
    num_demos = 1 if args.num_demos == 0 else args.num_demos
    positive_demos = generate_demonstrations(
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
    samples = []
    for _,fv,a in positive_demos:
        sample = [x for x in fv]+[str(a)]
        if sample not in samples:
            samples.append(sample)

    with open("samples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(samples)






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

    one_shot_learning(args=args)
    raise Exception('EXIT')



    num_demos = 1 if args.num_demos == 0 else args.num_demos
    positive_demos = generate_demonstrations(
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
        positive_demos = positive_demos[:1]
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
            epoch=epoch,
            use_known_error_envs=True
        )

        print(f"{sat = }")
        if not sat:
            obs, act = random_sampling_algorithm(positive_dict, trace)
            negative_dict[obs] = act
            print(f"Added Demonstration: {obs} -> {act}")
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
