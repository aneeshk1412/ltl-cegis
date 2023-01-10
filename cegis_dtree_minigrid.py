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
            idx = hashes[i + 1:].index(x) + i + 1
            return trace[:i], trace[i:idx]
        except ValueError:
            continue
    return trace, None


def train_and_save_model(positive_dict, negative_dict, extra_dict=None, seed=None):
    if extra_dict is None:
        assert set(positive_dict.keys()) & set(negative_dict.keys()) == set([])
        state_demos = pd.DataFrame(
            [state for state in positive_dict] +
            [state for state in negative_dict]
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
        new_act = random.sample(
            [a for a in ACT_KEY_TO_IDX.keys() if a != act], 1)
        return obs, new_act[0]
    return None, None


def random_loop_correction(positive_dict, trace):
    stem, loop = get_stem_and_loop(trace)
    if loop is None:
        return random_sampling_algorithm(positive_dict, stem)
    for _, obs, act in loop:
        if obs in positive_dict:
            continue
        new_act = random.sample(
            [a for a in ACT_KEY_TO_IDX.keys() if a != act], 1)
        return obs, new_act[0]
    return None, None


def one_shot_learning(args, score_type):
    # init variabales
    show_tree = False
    # how mnay attempts for finindg counter-examples? this is the size of the set of counter-examples returned for analysis
    number_of_counter_examples = 10

    # run demonstrations from the ground truth and save them to samples.csv file -- need not be called all the time
    # generate_and_save_samples(args)

    # load existing samples from samples.csv file (this file initially contains samples from 2 correct demonstrations)
    samples = []
    state_demos = []
    act_demos = []
    with open('samples.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for line in csv_reader:
            if '#' in line[0]:  # skip this sample since it is commented out
                continue
            samples.append(line)
            act_demos.append(line[-1])
            state_demos.append([eval(x) for x in line[:-1]])

    # train a BDT model based on the samples
    aspmodel = tree.DecisionTreeClassifier(
        class_weight="balanced", random_state=args.tree_seed)
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
    def action_selection_policy(env): return action_selection_policy_decision_tree(
        env, bdt_model, feature_register[args.env_name])

    # verify the learned ASP. Attempt to generate number_of_counter_examples many counter-examples and return them all. If a verification attempt is successful, i.e. no counter-examples are found, then return whatever counter-examples generated so far. If the first attempt is successful then we consider this ASP as fully verified. This can later change.
    sats, traces, epoch_score = verify_action_selection_policy(
        args.env_name,
        action_selection_policy,
        feature_register[args.env_name],
        seed=args.verifier_seed,
        num_trials=args.num_trials,
        timeout=args.timeout,
        show_window=args.show_window,
        tile_size=args.tile_size,
        agent_view=args.agent_view,
        use_known_error_envs=False,
        verify_each_step_manually=False,
        cex_count=number_of_counter_examples,
        score_type_in=score_type
    )
    print(f"{sats = }")

    # suggest a (set of) new samples based on the counter-examples (try to automate the insights from above step)
    if len(traces) > 0:
        # calculate the epoch score as the average number of epochs before a counter-example was found
        analyze_traces_suggest_repair(traces, epoch_score, score_type)
    else: # first attempt to verification was successful and no counter-example was found
        print('correct program is found')


def analyze_traces_suggest_repair(traces, epoch_score, score_type):
    # load known samples from samples.csv file (this is only used for manual analysis)
    # includes all samples (added manually not from demo) that are in the samples.csv file. Only samples which do not start with # are used for training
    manually_added_samples = []
    manually_added_actions = []
    # includes only samples that are actually used for training (samples starting with # are not used for training and are there simply for studying purposes)
    manually_added_samples_included_in_training = []
    with open('samples.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        begin = False
        for line in csv_reader:
            if '# begin samples from manual inspection of counter-examples' in line[0]:
                begin = True
                continue  # ignore samples from the demo
            if begin:  # samples from the demo are over and now we can read and store samples which are added manually
                sample = [eval(x.replace('#', '')) for x in line[:-1]]
                if '#' not in line[0]:
                    manually_added_samples_included_in_training.append(sample)
                manually_added_samples.append(sample)
                manually_added_actions.append(line[-1])

    # a dictionary to keep track of number of occurences of the manually added samples with repetition per trace
    # this dictionary shall contain the visited states (bv) as keys and the number of occurences as the value. Repetitions within each trace are counted
    mas_cnt_with_rep = {tuple(x): 0 for x in manually_added_samples}
    # similar to the above dictionary, however, this one counts at most 1 occurence of each state within a trace.
    mas_cnt_without_rep = {tuple(x): 0 for x in manually_added_samples}
    total_trace_lens = 0
    trace_cnt = 0
    event_map = dict()  # number of each state occurence in the given traces
    no_rep_event_map = dict()  # number of each state occurence in the given traces
    action_map = dict()
    for trace in traces:  # walk through traces
        trace_cnt += 1
        seen_bvs = set()
        for env, bv, a in trace:  # walk through each trace
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
            if bv in no_rep_event_map:
                no_rep_event_map[bv] = no_rep_event_map[bv] + 1
            else:
                no_rep_event_map[bv] = 1

    # following vars are used to find the max with and without repetition
    max_seen_cnt, max_index, chosen_bv = -1, -1, None
    no_rep_max_seen_cnt, no_rep_max_index, no_rep_chosen_bv = -1, -1, None
    res = [['feature'] + ['action taken', 'number w/ rep', 'number w/o rep'] +
           readable_headers_list()]  # this will bused later to print in tabular format
    i = -1
    for s in event_map:
        res.append(['bv#'+str(i)] + list(action_map[s]) + [event_map[s]
                                                           ] + [no_rep_event_map[s]] + [str(x)[0] for x in s])
        # identify the most frequent new (not used in training) state with and without repetition
        # we do not consider existing samples used for training to be picked for repair (i.e., we never roll back)
        if s in [tuple(x) for x in manually_added_samples_included_in_training]:
            i += 1
            continue
        if event_map[s] > max_seen_cnt:
            max_seen_cnt, max_index, chosen_bv = event_map[s], i, s
        if no_rep_event_map[s] > no_rep_max_seen_cnt:
            no_rep_max_seen_cnt, no_rep_max_index, no_rep_chosen_bv = no_rep_event_map[s], i, s
        i += 1
    transposed_data = list(zip(*res))

    print('\n\n\n')
    print('='*200)
    print('Analytics')
    print('='*200)
    print('\n')
    print('Known States (only for analytics):')
    print('----------------------------------')
    i = 0
    for bv in manually_added_samples:
        print(('sample#'+str(i)).ljust(10), str(mas_cnt_with_rep[tuple(bv)]).ljust(4, ' '),
              str(mas_cnt_without_rep[tuple(bv)]).ljust(4, ' '), str(bv in manually_added_samples_included_in_training).ljust(5, ' '), '  ',  str(manually_added_actions[i]).ljust(5, ' '), '  ',bv)
        i += 1
    print('')

    # print details about the given set of traces
    print('Trace Analytics:')
    print('----------------')
    print('* total len:   ', total_trace_lens)
    print('* trace count: ', trace_cnt)
    # epoch score = in total, how many attempts were made to find the given counter-examples? The higher score means it was harder for the model checker to find counter-examples, i.e. the quality of the ASP was higher
    print('* epoch accuracy: ', epoch_score)
    print()

    # print a summary of states encountered on traces and the frequency of each state in a table
    print('Observed States in Traces:')
    print('--------------------------')
    print(tabulate(transposed_data))
    print('-'*185)

    # print repair suggestions
    print('Repair Suggestions:')
    print('-------------------')
    known_tuples = [tuple(x) for x in manually_added_samples]
    # print ordered frequencies of states with repetition
    res = ''
    for s in reversed(sorted(event_map.values())):
        res += (str(s) + ', ')
    print('(w/  repetition) ordered frequencies:', res)
    print('(w/  repetition) suggested bv#'+str(max_index)+' to repair with',
          max_seen_cnt, 'repeitions:', str(chosen_bv).replace(' ', ''))
    print('(w/  repetition) index in known set:',
          known_tuples.index(chosen_bv) if chosen_bv in known_tuples else -1)
    print()
    # print ordered frequencies of states WITHOUT repetition
    res = ''
    for s in reversed(sorted(no_rep_event_map.values())):
        res += (str(s) + ', ')
    print('(w/o repetition) ordered frequencies:', res)
    print('(w/o repetition) suggested bv#'+str(no_rep_max_index)+' to repair with',
          str(no_rep_max_seen_cnt).ljust(3, ' '), 'repeitions:', str(no_rep_chosen_bv).replace(' ', ''))
    print('(w/o repetition) index in known set:', known_tuples.index(no_rep_chosen_bv)
          if no_rep_chosen_bv in known_tuples else -1)
   
    
    print ('\n')
    if trace_cnt > 1:
        print ('metric with no repetition is used since there are multiple counter-examples')
        final_chosen_bv = no_rep_chosen_bv
    else:
        print ('metric with repetition is used since there is only 1 counter-example')
        final_chosen_bv = chosen_bv

    print ('label epoch accuracies:')
    chosen_label = get_candidate_label_scores(args=args, new_sample_state=final_chosen_bv, test_cnt=30, score_type=score_type) #XXX 30 is hardcoded for now. make it a passed parameter # TODO
    
    #for label in label_scores:
    #    print ('                                        - '+label, label_scores[label])
    print('\n>> Suggested sample to add:     ', str(final_chosen_bv).replace(' ', '').replace(')','').replace('(','')+','+chosen_label)
    print('\n\n')
def get_pass_ratio_score():
    pass




def get_candidate_label_scores(args, new_sample_state, test_cnt, score_type):
    label_scores = {'left':-1, 'right':-1, 'up':-1, 'pageup':-1}
    for label in label_scores:
        state_demos = []
        act_demos = []
        with open('samples.csv', 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            for line in csv_reader:
                if '#' in line[0]:  # skip this sample since it is commented out
                    continue
                act_demos.append(line[-1])
                state_demos.append([eval(x) for x in line[:-1]])
        # add the new candidate sample to the training set
        state_demos.append(new_sample_state)
        act_demos.append(label)
        # train a BDT model based on the samples
        aspmodel = tree.DecisionTreeClassifier(class_weight="balanced", random_state=args.tree_seed)
        bdt_model = aspmodel.fit(state_demos, act_demos)
        def action_selection_policy(env): return action_selection_policy_decision_tree(env, bdt_model, feature_register[args.env_name])
        _, _, epoch_score = verify_action_selection_policy(
            args.env_name,
            action_selection_policy,
            feature_register[args.env_name],
            seed=args.verifier_seed,
            num_trials=args.num_trials,
            timeout=args.timeout,
            show_window=args.show_window,
            tile_size=args.tile_size,
            agent_view=args.agent_view,
            use_known_error_envs=False,
            verify_each_step_manually=False,
            cex_count=test_cnt,
            mute = True,
            score_type_in=score_type
        )
        label_scores[label] = epoch_score
        print ('    -'+label+": "+str(epoch_score))
    return max(label_scores, key=label_scores.get)
    
    




# This function needs to be called only once to generate a set of positive traces and store their state/action pairs in samples.csv file
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

    # eliminate duplicates
    samples = []
    for _, fv, a in positive_demos:
        sample = [x for x in fv]+[str(a)]  # append states and actions
        if sample not in samples:
            samples.append(sample)

    # dump the generated samples into a file
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
    #score_type = 'first_error_lens'
    score_type = 'pass_to_all_ratio'
    one_shot_learning(args=args, score_type=score_type)
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
        aspmodel = train_and_save_model(
            positive_dict, negative_dict, seed=args.tree_seed)

        def action_selection_policy(env): return action_selection_policy_decision_tree(
            env, aspmodel, feature_register[args.env_name])
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
            class_names=sorted(set(positive_dict.values()) |
                               set(negative_dict.values())),
            label="none",
            precision=1,
            feature_names=header_register[args.env_name],
            rounded=True,
            fontsize=5,
            proportion=True,
        )
        plt.show()
