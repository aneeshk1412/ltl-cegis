#!/usr/bin/env python3

from typing import List, Dict, Callable, Set
from copy import deepcopy
from random import Random
from collections import deque

from trace_minigrid import Trace, State
from graph_minigrid import TransitionGraph
from learner_minigrid import train_policy, plot_policy
from verifier_minigrid import verify_policy, simulate_policy_on_list_of_envs
from utils import (
    csv_to_positive_samples_dict,
    pickle_to_demo_traces,
    is_sample_present,
    debug,
)
from commons import (
    get_augmented_policy,
    get_new_working_and_other_traces,
    check_conflicts,
    resolve_conflicts,
    get_arguments,
)

from minigrid.core.constants import ACT_SET
from minigrid.minigrid_env import MiniGridEnv


def correct_single_trace(
    trace: Trace,
    policy: Callable[[MiniGridEnv], str],
    decided_samples: Dict[State, str],
    env_name: str,
    graph: TransitionGraph,
    all_states: Set[State],
):
    env, _, _, _, _ = trace[0]
    trace_queue = deque([(trace, dict())])
    num_traces = 1
    graph.add_trace(trace)
    for _, s, _, _, s_n in trace.get_abstract_trace():
        all_states.add(s)
        all_states.add(s_n)

    while len(trace_queue):
        current_trace, current_ss = trace_queue.popleft()
        if all(
            is_sample_present(decided_samples, s, a)
            or is_sample_present(current_ss, s, a)
            for _, s, a, _, _ in current_trace.get_abstract_trace()
        ):
            ## This ss is unsatisfiable
            continue
        for e, s, a, _, _ in current_trace.get_abstract_trace():
            if s in decided_samples:
                assert decided_samples[s] == a
                continue
            if s in current_ss:
                assert current_ss[s] == a
                continue

            for a_p in sorted(ACT_SET - set([a])):
                new_ss = dict((si, ai) for si, ai in current_ss.items())
                new_ss[s] = a_p
                new_policy = get_augmented_policy(policy, new_ss, env_name)

                sat, sat_trace_pairs = simulate_policy_on_list_of_envs(
                    env_name=env_name,
                    env_list=[e],
                    policy=new_policy,
                    show_window=False,
                )
                graph.add_trace(sat_trace_pairs[0][1])
                for _, s, _, _, s_n in sat_trace_pairs[0][1].get_abstract_trace():
                    all_states.add(s)
                    all_states.add(s_n)
                if sat:
                    print(f"Traces seen before correcting this trace: {num_traces}")
                    print()
                    sat, sat_trace_pairs = simulate_policy_on_list_of_envs(
                        env_name=env_name,
                        env_list=[env],
                        policy=new_policy,
                        show_window=False,
                    )
                    for _, s, a, _, _ in sat_trace_pairs[0][1]:
                        if s in decided_samples:
                            assert decided_samples[s] == a
                            continue
                        new_ss[s] = a
                    return new_ss

                num_traces += 1
                debug(f"Number of Traces till now {num_traces}", end="\r")
                trace_queue.append((sat_trace_pairs[0][1], deepcopy(new_ss)))
    return None


if __name__ == "__main__":
    args = get_arguments()
    verifier_rng = Random(args.verifier_seed)

    positive_demos = pickle_to_demo_traces(env_name=args.env_name)
    decided_samples = csv_to_positive_samples_dict(env_name=args.env_name)
    speculated_samples = dict()

    working_traces: List[Trace] = list()
    other_traces: List[Trace] = list()

    all_states = set()
    for s in decided_samples:
        all_states.add(s)

    graph = TransitionGraph(env_name=args.env_name)
    graph.add_traces(positive_demos)

    epoch = 0

    while True:
        policy, _ = train_policy(
            env_name=args.env_name,
            decided_samples=decided_samples,
            speculated_samples=speculated_samples,
            seed=args.learner_seed,
            save=False,
        )
        sat, traces = verify_policy(
            env_name=args.env_name,
            policy=policy,
            seed=verifier_rng.randrange(int(1e10)),
            num_rruns=args.num_rruns,
            max_steps=args.max_steps,
            use_saved_envs=False,
            show_window=args.show_window,
        )
        if sat:
            sat, traces = verify_policy(
                env_name=args.env_name,
                policy=policy,
                seed=verifier_rng.randrange(int(1e10)),
                num_rruns=300,
                max_steps=args.max_steps,
                use_saved_envs=False,
                show_window=args.show_window,
            )
            if sat:
                break

        """ Main Algorithm starts here """

        working_traces, other_traces = get_new_working_and_other_traces(
            working_traces, other_traces, policy, traces
        )
        suggested_samples = correct_single_trace(
            working_traces[0],
            policy=policy,
            decided_samples=decided_samples,
            env_name=args.env_name,
            graph=graph,
            all_states=all_states,
        )
        if suggested_samples is None:
            debug(f"Number of Demo States: {len(decided_samples)}")
            debug(
                f"Number of New States Seen: {len(all_states) - len(decided_samples)}"
            )
            raise Exception("UNSAT: Could not come up with a Suggested Sample set")
        if check_conflicts(speculated_samples, suggested_samples, args.env_name):
            suggested_samples = resolve_conflicts(
                speculated_samples,
                suggested_samples,
                policy,
                args.env_name,
                working_traces,
                other_traces,
            )
            if speculated_samples is None:
                debug(f"Number of Demo States: {len(decided_samples)}")
                debug(
                    f"Number of New States Seen: {len(all_states) - len(decided_samples)}"
                )
                raise Exception("Both Speculation Sets failed some Traces")
        speculated_samples.update(suggested_samples)
        debug(
            f"Total Number of New States Seen: {len(all_states) - len(decided_samples)}"
        )
        epoch += 1
        graph.show_graph(name="bfs.html")

    print(f"Epochs to Completion: {epoch}")
    print(f"Total Number of Demo States: {len(decided_samples)}")
    print(f"Total Number of New States Seen: {len(all_states) - len(decided_samples)}")

    _, model = train_policy(
        env_name=args.env_name,
        decided_samples=decided_samples,
        speculated_samples=speculated_samples,
        seed=args.learner_seed,
        save=True,
    )
    if args.plot_policy:
        class_names = sorted(
            set(decided_samples.values()) | set(speculated_samples.values())
        )
        plot_policy(model=model, class_names=class_names, env_name=args.env_name)
