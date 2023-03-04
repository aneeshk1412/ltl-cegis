from typing import Callable

from utils import state_to_pretty_string
from trace_minigrid import Trace
from dsl_minigrid import feature_register
from verifier_minigrid import verify_policy_on_envs

from minigrid.minigrid_env import MiniGridEnv


def get_augmented_policy(policy, speculation_set, env_name):
    def augmented_policy(env):
        state = feature_register[env_name](env)
        if state in speculation_set:
            return speculation_set[state]
        else:
            return policy(env)

    return augmented_policy


def check_if_policy_runs_trace(
    policy: Callable[[MiniGridEnv], str],
    trace: Trace,
) -> bool:
    return all(policy(env) == a for env, _, a, _, _ in trace)


def get_new_working_and_other_traces(working_traces, other_traces, policy, new_traces):
    work_to_correct = [
        trace
        for trace in working_traces
        if not check_if_policy_runs_trace(policy, trace)
    ]
    correct_to_work = [
        trace for trace in other_traces if check_if_policy_runs_trace(policy, trace)
    ]
    if correct_to_work:
        print(
            f"Number of traces moved from Corrected to Working: {len(correct_to_work)}"
        )

    new_working_traces = (
        [trace for trace in working_traces if check_if_policy_runs_trace(policy, trace)]
        + correct_to_work
        + new_traces
    )
    new_other_traces = work_to_correct + [
        trace for trace in other_traces if not check_if_policy_runs_trace(policy, trace)
    ]
    return new_working_traces, new_other_traces


def check_conflicts(speculated_samples, suggested_samples, env_name):
    flag = False
    okay_set = {"left", "right"}
    for s in set(suggested_samples.keys()) & set(speculated_samples.keys()):
        if suggested_samples[s] != speculated_samples[s]:
            if suggested_samples[s] in okay_set and speculated_samples[s] in okay_set:
                continue
            caption = state_to_pretty_string(s, env_name)
            print(f"State: {caption}")
            print(f"Unmatched : {suggested_samples[s] = }, {speculated_samples[s] = }")
            flag = True
    return flag


def resolve_conflicts(
    speculated_samples,
    suggested_samples,
    policy,
    env_name,
    working_traces,
    other_traces,
):
    print(suggested_samples)
    ss_A = dict()
    for s in list(speculated_samples.keys()) + list(suggested_samples.keys()):
        if s in speculated_samples:
            ss_A[s] = speculated_samples[s]
        else:
            ss_A[s] = suggested_samples[s]

    ss_B = dict()
    for s in list(speculated_samples.keys()) + list(suggested_samples.keys()):
        if s in suggested_samples:
            ss_B[s] = suggested_samples[s]
        else:
            ss_B[s] = speculated_samples[s]

    policy_A = get_augmented_policy(policy, ss_A, env_name)
    policy_B = get_augmented_policy(policy, ss_B, env_name)

    sat_A, sat_trace_pairs_A = verify_policy_on_envs(
        env_name=env_name,
        env_list=[t[0][0] for t in working_traces] + [t[0][0] for t in other_traces],
        policy=policy_A,
        show_window=False,
    )

    sat_B, sat_trace_pairs_B = verify_policy_on_envs(
        env_name=env_name,
        env_list=[t[0][0] for t in working_traces] + [t[0][0] for t in other_traces],
        policy=policy_B,
        show_window=False,
    )

    if sat_A and sat_B:
        print("Both work")
        return ss_B
    if sat_A:
        print("Only Old works in all")
        return ss_A
    if sat_B:
        print("Only New works in all")
        return ss_B

    print("Failing envs A")
    for i, (s, t) in enumerate(sat_trace_pairs_A):
        if not s:
            _, _ = verify_policy_on_envs(
                env_name=env_name,
                env_list=[t[0][0]],
                policy=policy_A,
                show_window=True,
                block=True,
            )
            _, _ = verify_policy_on_envs(
                env_name=env_name,
                env_list=[t[0][0]],
                policy=policy,
                show_window=True,
                block=True,
            )
    print("Failing envs B")
    for i, (s, t) in enumerate(sat_trace_pairs_B):
        if not s:
            _, _ = verify_policy_on_envs(
                env_name=env_name,
                env_list=[t[0][0]],
                policy=policy_B,
                show_window=True,
                block=True,
            )
            _, _ = verify_policy_on_envs(
                env_name=env_name,
                env_list=[t[0][0]],
                policy=policy,
                show_window=True,
                block=True,
            )
    return None


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
