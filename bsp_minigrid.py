#!/usr/bin/env python3
import z3
from random import Random
from typing import List, Dict, Set

from trace_minigrid import Trace, State
from graph_minigrid import TransitionGraph
from learner_minigrid import train_policy, plot_policy
from verifier_minigrid import verify_policy
from utils import (
    csv_to_positive_samples_dict,
    pickle_to_demo_traces,
    state_to_bitstring,
    debug,
)

from minigrid.core.constants import ACT_SET


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
        "--z3-seed",
        type=int,
        help="random seed for z3",
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


def sample_to_name(s: State, a: str):
    return state_to_bitstring(s) + "_" + a


def add_decided_condition(
    decided_samples: Dict[State, str],
    name_to_z3bool: Dict[str, z3.Bool],
    solver: z3.Solver,
):
    solver.add(
        *[name_to_z3bool[sample_to_name(s, a)] for s, a in decided_samples.items()]
    )


def print_decided_condition(
    decided_samples: Dict[State, str],
    name_to_z3bool: Dict[str, z3.Bool],
):
    for s, a in decided_samples.items():
        debug(name_to_z3bool[sample_to_name(s, a)])


def add_traces_condition(
    all_traces: List[Trace],
    decided_samples: Dict[State, str],
    name_to_z3bool: Dict[str, z3.Bool],
    solver: z3.Solver,
):
    solver.add(
        *[
            z3.Or(
                [
                    z3.Not(name_to_z3bool[sample_to_name(s, a)])
                    for _, s, a, _, _ in trace.get_abstract_trace()
                    if s not in decided_samples
                ]
            )
            for trace in all_traces
        ]
    )


def print_traces_condition(
    all_traces: List[Trace],
    decided_samples: Dict[State, str],
    name_to_z3bool: Dict[str, z3.Bool],
):
    for trace in all_traces:
        debug(
            z3.Or(
                [
                    z3.Not(name_to_z3bool[sample_to_name(s, a)])
                    for _, s, a, _, _ in trace.get_abstract_trace()
                    if s not in decided_samples
                ]
            )
        )


def add_exactly_one_condition(
    all_states: Set[State],
    name_to_z3bool: Dict[str, z3.Bool],
    solver: z3.Solver,
):
    solver.add(
        *[
            z3.PbEq([(name_to_z3bool[sample_to_name(s, a)], 1) for a in ACT_SET], 1)
            for s in all_states
        ]
    )


def print_exactly_one_condition(
    all_states: Set[State],
    name_to_z3bool: Dict[str, z3.Bool],
):
    for s in all_states:
        debug(z3.PbEq([(name_to_z3bool[sample_to_name(s, a)], 1) for a in ACT_SET], 1))


if __name__ == "__main__":
    args = get_arguments()
    verifier_rng = Random(args.verifier_seed)

    positive_demos = pickle_to_demo_traces(env_name=args.env_name)
    decided_samples = csv_to_positive_samples_dict(env_name=args.env_name)
    speculated_samples = dict()

    all_traces: List[Trace] = list()

    graph = TransitionGraph(env_name=args.env_name)
    graph.add_traces(positive_demos)

    epoch = 0

    z3.set_option("smt.random_seed", args.z3_seed)
    name_to_sample = dict()
    name_to_z3bool = dict()
    all_states = set()

    for s in decided_samples:
        if s not in all_states:
            all_states.add(s)
            for a in ACT_SET:
                name = sample_to_name(s, a)
                name_to_sample[name] = (s, a)
                name_to_z3bool[name] = z3.Bool(name)

    while True:
        debug(f"Start of Epoch: {epoch}")
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
        all_traces.extend(traces)
        print(len(traces))
        graph.add_trace(traces[0])

        for trace in traces:
            for _, s, _, _, _ in trace:
                if s not in all_states:
                    all_states.add(s)
                    for a in ACT_SET:
                        name = sample_to_name(s, a)
                        name_to_sample[name] = (s, a)
                        name_to_z3bool[name] = z3.Bool(name)

        solver = z3.Solver()
        add_traces_condition(all_traces, decided_samples, name_to_z3bool, solver)
        add_decided_condition(decided_samples, name_to_z3bool, solver)
        add_exactly_one_condition(all_states, name_to_z3bool, solver)

        if solver.check() != z3.sat:
            print_traces_condition(all_traces, decided_samples, name_to_z3bool)
            print_decided_condition(decided_samples, name_to_z3bool)
            print_exactly_one_condition(all_states, name_to_z3bool)

            debug(f"Total Number of Demo States: {len(decided_samples)}")
            debug(
                f"Total Number of New States Seen: {len(all_states) - len(decided_samples)}"
            )

            raise Exception("UNSAT: Could not come up with a Speculated Sample set")

        assignment = solver.model()
        speculated_samples = dict()
        for s in all_states:
            for a in ACT_SET:
                if assignment[name_to_z3bool[sample_to_name(s, a)]]:
                    if s in decided_samples:
                        assert decided_samples[s] == a
                        continue
                    if s in speculated_samples:
                        raise Exception("Odd, 2 actions for same state")
                    speculated_samples[s] = a
        debug(f"End of Epoch: {epoch}")
        debug(f"Total Number of New States Seen: {len(all_states) - len(decided_samples)}")
        epoch += 1
        graph.show_graph(name='bsp.html')

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
