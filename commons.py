from copy import deepcopy
from typing import Callable

from utils import state_to_pretty_string, debug, state_to_bitstring
from trace_minigrid import Trace
from dsl_minigrid import feature_register
from verifier_minigrid import simulate_policy_on_list_of_envs, simulate_policy_on_env
from graph_minigrid import TransitionGraph

from minigrid.minigrid_env import MiniGridEnv


def add_trace_to_graph_and_all_states(trace, graph, all_states):
    graph.add_trace(trace)
    for _, s, _, _, s_n in trace.get_abstract_trace():
        all_states.add(s)
        all_states.add(s_n)


def inverse(a):
    if a == "left":
        return "right"
    if a == "right":
        return "left"
    if a == "pageup":
        return "pagedown"
    if a == "pagedown":
        return "pageup"


def add_inverse_actions_to_states(new_ss, all_states, decided_samples, graph):
    while True:
        inv_ss = dict()
        for s in all_states:
            if s in decided_samples:
                continue
            if s in new_ss:
                continue
            for s_p in new_ss:
                for a in {"left", "right", "pageup", "pagedown"}:
                    if graph.graph.has_edge(
                        state_to_bitstring(s_p),
                        state_to_bitstring(s),
                        key=a,
                    ):
                        inv_ss[s] = inverse(a)
        if not inv_ss:
            break
        for s in inv_ss:
            new_ss[s] = inv_ss[s]
    return new_ss


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
        debug(
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
    for s in set(suggested_samples.keys()) & set(speculated_samples.keys()):
        if suggested_samples[s] != speculated_samples[s]:
            caption = state_to_pretty_string(s, env_name)
            debug(f"Unmatched : {suggested_samples[s] = }, {speculated_samples[s] = }")
            debug(f"State: {caption}")
            flag = True
    debug("\n\n")
    return flag


def resolve_conflicts(
    speculated_samples,
    suggested_samples,
    policy,
    env_name,
    working_traces,
    other_traces,
    block=False,
):
    policy_old_ss = get_augmented_policy(policy, speculated_samples, env_name)
    sat_old, sat_trace_pairs_old = simulate_policy_on_list_of_envs(
        env_name=env_name,
        env_list=[t[0][0] for t in working_traces] + [t[0][0] for t in other_traces],
        policy=policy_old_ss,
        show_window=False,
    )

    ss_rewrite = deepcopy(speculated_samples)
    for s, a in suggested_samples.items():
        ss_rewrite[s] = a
    policy_rewrite = get_augmented_policy(policy, ss_rewrite, env_name)
    sat_rewrite, sat_trace_pairs_rewrite = simulate_policy_on_list_of_envs(
        env_name=env_name,
        env_list=[t[0][0] for t in working_traces] + [t[0][0] for t in other_traces],
        policy=policy_rewrite,
        show_window=False,
    )

    if sat_old and sat_rewrite:
        raise Exception("Both Work, weird")
    elif sat_old:
        raise Exception("Old Works, weird")
    elif sat_rewrite:
        debug("New Works")
        return ss_rewrite
    else:
        return ss_rewrite

    combined_list = [t[0][0] for t in working_traces] + [t[0][0] for t in other_traces]

    debug("Old Speculated Samples:\n")
    for s, a in speculated_samples.items():
        debug("State:")
        debug(state_to_pretty_string(s, env_name))
        debug(f"Action: {a}")
    debug()
    debug()

    debug("Rewritten Samples:\n")
    for s, a in ss_rewrite.items():
        debug("State:")
        debug(state_to_pretty_string(s, env_name))
        debug(f"Action: {a}")
    debug()
    debug()

    conflicts = set()
    for s, a in speculated_samples.items():
        if s in suggested_samples and suggested_samples[s] != a:
            conflicts.add(s)

    g = TransitionGraph(env_name)
    for i, ((s_old, trace_old), (s_rewrite, trace_rewrite)) in enumerate(
        zip(sat_trace_pairs_old, sat_trace_pairs_rewrite)
    ):
        print(i, "Working Trace" if i < len(working_traces) else "Other Traces")
        print(f"{s_old = } {s_rewrite = }")
        if s_old and not s_rewrite:
            _, _ = simulate_policy_on_env(
                env_name=env_name,
                env=combined_list[i],
                policy=policy_old_ss,
                show_window=True,
                block=block,
                detect_collisions=False,
            )
            for e, s, a, e_n, s_n in trace_old:
                if s in conflicts:
                    label = f"Env{i} P_1 on Conflict"
                elif s in speculated_samples:
                    label = f"Env{i} P_1 Speculated"
                else:
                    label = f"Env{i} P_1 Generalized"
                g.add_transition((e, s, a, e_n, s_n), "demo", label=label)
            _, _ = simulate_policy_on_env(
                env_name=env_name,
                env=combined_list[i],
                policy=policy_rewrite,
                show_window=True,
                block=block,
                detect_collisions=False,
            )
            for e, s, a, e_n, s_n in trace_rewrite:
                if s in conflicts:
                    label = f"Env{i} P_2 on Conflict"
                elif s in suggested_samples:
                    label = f"Env{i} P_2 new Suggestions"
                elif s in speculated_samples:
                    label = f"Env{i} P_2 Speculated"
                else:
                    label = f"Env{i} P_2 Generalized"
                g.add_transition((e, s, a, e_n, s_n), "cex", label=label)
        if s_rewrite and not s_old:
            _, _ = simulate_policy_on_env(
                env_name=env_name,
                env=combined_list[i],
                policy=policy_old_ss,
                show_window=True,
                block=block,
                detect_collisions=False,
            )
            for e, s, a, e_n, s_n in trace_old:
                if s in conflicts:
                    label = f"Env{i} P_1 on Conflict"
                elif s in speculated_samples:
                    label = f"Env{i} P_1 Speculated"
                else:
                    label = f"Env{i} P_1 Generalized"
                g.add_transition((e, s, a, e_n, s_n), "cex", label=label)
            _, _ = simulate_policy_on_env(
                env_name=env_name,
                env=combined_list[i],
                policy=policy_rewrite,
                show_window=True,
                block=block,
                detect_collisions=False,
            )
            for e, s, a, e_n, s_n in trace_rewrite:
                if s in conflicts:
                    label = f"Env{i} P_2 on Conflict"
                elif s in suggested_samples:
                    label = f"Env{i} P_2 new Suggestions"
                elif s in speculated_samples:
                    label = f"Env{i} P_2 Speculated"
                else:
                    label = f"Env{i} P_2 Generalized"
                g.add_transition((e, s, a, e_n, s_n), "demo", label=label)
    g.show_graph("witness.html")
    raise Exception("Whoops")


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
