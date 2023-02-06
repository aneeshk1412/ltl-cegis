#!/usr/bin/env python3

import pickle
import random
from typing import List
from collections import defaultdict

from minigrid.core.constants import ACT_KEY_TO_IDX

from graph_utils import Trace
from utils import csv_to_positive_samples_dict
from learner_minigrid import train_policy, plot_policy
from policy_minigrid import policy_decision_tree, feature_register
from verifier_minigrid import verify_policy
from dsl_minigrid import dot


def satisfies(policy, trace):
    return all(policy(env) == a for env, _, a, _, _ in trace)


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

    untried_samples = defaultdict(lambda : set(ACT_KEY_TO_IDX.keys()))

    epoch = 0

    while True:
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
            use_saved_envs=True,
            show_window=args.show_window,
        )
        if sat:
            break

        """ Main Algorithm starts here """

        working_traces, corrected_traces = get_new_working_and_corrected_traces(
            working_traces, corrected_traces, policy, traces
        )

        # set_cover = defaultdict(set)
        for i, trace in enumerate(working_traces):
            loop = trace.get_loop()
            if loop is None:
                continue

            loop_changeable = []
            for t in loop:
                _, s, a, _, _ = t
                if s in decided_samples and decided_samples[s] == a:
                    continue
                if s in decided_samples:
                    speculated_samples[s] = a
                    continue
                loop_changeable.append(t)
            print(f"{len(loop_changeable) = }")
            print()
            ''' If len == 1 Invariant or a Loop with single changeable state due to Demos '''

            for t in random.sample(loop_changeable, 1):
                _, s, a, _, _ = loop_changeable[0]

                ''' Proxy for other states with nearest satisfying formula '''
                max_dp, max_d_s = 0, None
                for d_s in decided_samples:
                    if dot(d_s, s) > max_dp:
                        max_dp = dot(d_s, s)
                        max_d_s = d_s
                if max_d_s:
                    speculated_samples[s] = decided_samples[d_s]
                else:
                    speculated_samples[s] = random.sample(list(untried_samples[s]), 1)[0]

                if speculated_samples[s] in untried_samples[s]:
                    untried_samples[s].remove(speculated_samples[s])

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
