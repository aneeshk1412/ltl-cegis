#!/usr/bin/env python3

import pickle
from typing import List
from copy import deepcopy
from collections import defaultdict, deque

from minigrid.core.constants import ACT_SET

from graph_utils import Trace
from utils import csv_to_positive_samples_dict, intersperse
from learner_minigrid import train_policy, plot_policy
from policy_minigrid import policy_decision_tree, feature_register, header_register
from verifier_minigrid import verify_policy, verify_policy_on_envs


def satisfies(policy, trace):
    return all(policy(env) == a for env, _, a, _, _ in trace)


def correct_single_trace(trace: Trace, policy, decided_samples, speculated_samples, env_name):
    env, _, _, _, _ = trace[0]
    trace_queue = deque([(trace, dict())])
    num_traces = 1
    # print(f"{len(trace.get_abstract_trace()) = } {len(trace.get_abstract_loop()) = } {len(trace.get_abstract_stem()) = }")
    while len(trace_queue):
        current_trace, current_ss = trace_queue.popleft()
        cond = lambda d, s, a: s in d and d[s] == a
        if all(cond(decided_samples, s, a) or cond(current_ss, s, a) or cond(speculated_samples, s, a) for _, s, a, _, _ in current_trace.get_abstract_trace()):
            ## This ss is unsatisfiable
            continue
        for e, s, a, _, _ in current_trace.get_abstract_trace():
            if s in decided_samples:
                assert decided_samples[s] == a
                continue
            if s in speculated_samples:
                assert speculated_samples[s] == a
                continue
            if s in current_ss:
                assert current_ss[s] == a
                continue

            for a_p in sorted(ACT_SET - set([a])):
                new_ss = dict((si, ai) for si, ai in current_ss.items())
                new_ss[s] = a_p
                new_policy = lambda env: new_ss[feature_register[env_name](env)] if feature_register[env_name](env) in new_ss else policy(env)
                # print(f"{new_ss}")
                sat, sat_trace_pairs = verify_policy_on_envs(
                    env_name=env_name,
                    env_list=[e],
                    policy=new_policy,
                    show_window=False,
                )
                if sat:
                    print(f"{num_traces}")
                    # print(f"Number of Traces added before correcting this trace: {num_traces}")
                    # print()
                    sat, sat_trace_pairs = verify_policy_on_envs(
                        env_name=env_name,
                        env_list=[env],
                        policy=new_policy,
                        show_window=False,
                    )
                    for _, s, a, _, _ in sat_trace_pairs[0][1]:
                        if not s in decided_samples and not s in speculated_samples:
                            new_ss[s] = a
                    return new_ss

                # print(f"Number of Traces till now {num_traces}", end='\r')
                num_traces += 1
                trace_queue.append((sat_trace_pairs[0][1], deepcopy(new_ss)))
    print("UNSAT")


def get_new_working_and_corrected_traces(
    working_traces, corrected_traces, policy, new_traces
):
    print(f"Number of new traces: {len(new_traces)}")

    work_to_correct = [
        trace for trace in working_traces if not satisfies(policy, trace)
    ]
    correct_to_work = [trace for trace in corrected_traces if satisfies(policy, trace)]
    print(f"Number of traces moved from Working to Corrected: {len(work_to_correct)}")
    print(f"Number of traces moved from Corrected to Working: {len(correct_to_work)}")

    new_working_traces = (
        [trace for trace in working_traces if satisfies(policy, trace)]
        + correct_to_work
        + new_traces
    )
    new_corrected_traces = work_to_correct + [
        trace for trace in corrected_traces if not satisfies(policy, trace)
    ]
    print(f"{len(new_working_traces) = } {len(new_corrected_traces) = }")
    print()
    return new_working_traces, new_corrected_traces


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        help="gym environment to load",
        default="MiniGrid-DoorKey-16x16-v0",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="max steps to complete the task",
        default=100,
    )
    parser.add_argument(
        "--verifier-seed",
        type=int,
        help="random seed for the model checker",
        default=None,
    )
    parser.add_argument(
        "--num-rruns",
        type=int,
        help="number of random runs to verify on",
        default=100,
    )
    """ TODO: Currently the learner might produce different decision trees
        for the same training data. So the learner has been seeded.
    """
    parser.add_argument(
        "--learner-seed",
        type=int,
        help="random seed for the learning algorithm",
        default=100,
    )
    parser.add_argument(
        "--plot-policy",
        default=False,
        help="whether to show the policy tree",
        action="store_true",
    )
    parser.add_argument(
        "--show-window",
        default=False,
        help="whether to show the animation window",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    with open(args.env_name + "-demos.pkl", "rb") as f:
        positive_demos = pickle.load(f)

    decided_samples = csv_to_positive_samples_dict(
        filename=args.env_name + "-demos.csv", env_name=args.env_name
    )
    speculated_samples = dict()

    working_traces : List[Trace] = list()
    corrected_traces : List[Trace] = list()

    possible_actions_for_state = defaultdict(lambda : set(act for act in ACT_SET))

    epoch = 0

    while True:
        working_traces.sort(key=lambda x: len(x.get_loop()))
        policy_model = train_policy(
            decided_samples=decided_samples,
            speculated_samples=speculated_samples,
            seed=args.learner_seed,
            save=False,
        )
        policy = lambda env: policy_decision_tree(
            env, policy_model, feature_register[args.env_name]
        )
        sat, traces = verify_policy(
            env_name=args.env_name,
            policy=policy,
            seed=args.verifier_seed,
            num_rruns=args.num_rruns,
            max_steps=args.max_steps,
            use_saved_envs=False,
            show_window=args.show_window,
        )
        if sat:
            sat, traces = verify_policy(
                env_name=args.env_name,
                policy=policy,
                seed=args.verifier_seed,
                num_rruns=100,
                max_steps=args.max_steps,
                use_saved_envs=False,
                show_window=args.show_window,
            )
            if sat:
                break

        """ Main Algorithm starts here """

        working_traces, corrected_traces = get_new_working_and_corrected_traces(
            working_traces, corrected_traces, policy, traces
        )
        suggested_samples = correct_single_trace(working_traces[0], policy=policy, decided_samples=decided_samples, speculated_samples=speculated_samples, env_name=args.env_name)
        flag = False
        for s in set(suggested_samples.keys()) & set(speculated_samples.keys()):
            if suggested_samples[s] != speculated_samples[s]:
                if suggested_samples[s] in {'left', 'right'} and speculated_samples[s] in {'left', 'right'}:
                    continue
                caption = "    ".join(
                    intersperse(
                        [header_register[args.env_name][i] for i in range(len(s)) if s[i]], "\n", 2
                    )
                )
                print(f"State: {caption}")
                print(f"Unmatched : {suggested_samples[s] = }, {speculated_samples[s] = }")
                flag = True
        if flag:
            raise Exception("Newly Suggested Samples differ from Speculated Samples till the Last Iteration")
        speculated_samples.update(suggested_samples)
        epoch += 1

    print(f"Epochs to Completion: {epoch}")

    policy_model = train_policy(
        decided_samples=decided_samples,
        speculated_samples=speculated_samples,
        seed=args.learner_seed,
        save=True,
    )
    if args.plot_policy:
        class_names = sorted(
            set(decided_samples.values()) | set(speculated_samples.values())
        )
        plot_policy(
            policy=policy_model, class_names=class_names, env_name=args.env_name
        )
