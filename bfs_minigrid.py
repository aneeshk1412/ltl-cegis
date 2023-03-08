#!/usr/bin/env python3

from typing import List, Dict, Callable, Set
from copy import deepcopy
from random import Random
from collections import deque

from trace_minigrid import Trace, State
from graph_minigrid import TransitionGraph
from learner_minigrid import train_policy, plot_policy
from verifier_minigrid import verify_policy, simulate_policy_on_env
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
    add_trace_to_graph_and_all_states,
    add_inverse_actions_to_states,
)

from minigrid.core.constants import ACT_SET
from minigrid.minigrid_env import MiniGridEnv


def correct_single_trace(
    trace: Trace,
    policy: Callable[[MiniGridEnv], str],
    decided_samples: Dict[State, str],
    speculated_samples: Dict[State, str],
    env_name: str,
    graph: TransitionGraph,
    all_states: Set[State],
):
    env, _, _, _, _ = trace[0]
    trace_queue = deque([(trace, deepcopy(speculated_samples))])
    num_traces = 1

    add_trace_to_graph_and_all_states(trace, graph, all_states)

    while len(trace_queue):
        current_trace, current_ss = trace_queue.popleft()
        assert set(current_ss.keys()) & set(decided_samples.keys()) == set([])

        if all(
            is_sample_present(decided_samples, s, a)
            or is_sample_present(current_ss, s, a)
            for _, s, a, _, _ in current_trace.get_abstract_trace()
        ):
            ## This ss is unsatisfiable
            continue

        for e, s, a, _, _ in reversed(current_trace.get_abstract_trace()):
            if s in decided_samples:
                assert decided_samples[s] == a
                continue
            if s in current_ss:
                assert current_ss[s] == a
                continue

            for a_p in sorted(ACT_SET - set([a])):
                new_ss = deepcopy(current_ss)
                new_ss[s] = a_p
                new_policy = get_augmented_policy(policy, new_ss, env_name)

                sim_sat, sim_trace = simulate_policy_on_env(
                    env_name=env_name,
                    env=e,
                    policy=new_policy,
                    show_window=False,
                )
                add_trace_to_graph_and_all_states(trace, graph, all_states)

                if sim_sat:
                    print(f"Traces seen before correcting this trace: {num_traces}")
                    fin_sat, fin_trace = simulate_policy_on_env(
                        env_name=env_name,
                        env=env,
                        policy=new_policy,
                        show_window=args.show_window,
                    )
                    assert fin_sat
                    add_trace_to_graph_and_all_states(fin_trace, graph, all_states)
                    for _, s, a, _, _ in fin_trace.get_abstract_trace():
                        if s in decided_samples:
                            assert s not in new_ss
                            continue
                        new_ss[s] = a
                    debug(f"Size of Speculation: {len(new_ss)}")
                    new_ss = add_inverse_actions_to_states(
                        new_ss, all_states, decided_samples, graph
                    )
                    debug(f"Size of Speculation after Inverse Semantics: {len(new_ss)}")
                    return new_ss

                num_traces += 1
                debug(f"Traces till now {num_traces}", end="\r")
                trace_queue.append((sim_trace, deepcopy(new_ss)))
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
            speculated_samples=speculated_samples,
            env_name=args.env_name,
            graph=graph,
            all_states=all_states,
        )

        if suggested_samples is None:
            _, _ = simulate_policy_on_env(
                env_name=args.env_name,
                env=working_traces[0][0][0],
                policy=policy,
                max_steps=args.max_steps,
                show_window=True,
                detect_collisions=False,
                block=True,
            )
            debug(f"Demo States: {len(decided_samples)}")
            debug(f"New States Seen: {len(all_states) - len(decided_samples)}")
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
            if suggested_samples is None:
                debug(f"Demo States: {len(decided_samples)}")
                debug(f"New States Seen: {len(all_states) - len(decided_samples)}")
                raise Exception("Both Speculation Sets failed some Traces")

        speculated_samples.update(suggested_samples)
        debug(f"New States Seen: {len(all_states) - len(decided_samples)}")
        epoch += 1
        print()
        graph.show_graph(name="bfs.html")

    print(f"Epochs to Completion: {epoch}")
    print(f"Total Demo States: {len(decided_samples)}")
    print(f"Total New States Seen: {len(all_states) - len(decided_samples)}")

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
