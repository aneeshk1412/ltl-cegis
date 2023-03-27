#!/usr/bin/env python3
import z3
from random import Random
from copy import deepcopy
from typing import List, Dict, Set

from trace_minigrid import Trace, State
from graph_minigrid import TransitionGraph
from learner_minigrid import train_policy, plot_policy
from verifier_minigrid import verify_policy, simulate_policy_on_list_of_envs
from utils import (
    csv_to_positive_samples_dict,
    pickle_to_demo_traces,
    state_to_bitstring,
    state_to_string,
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


def add_singleton_constraint(solver: z3.Solver, vars: List[z3.Bool]):
    solver.add(z3.And(z3.AtMost(*vars, 1), z3.Or(vars)))


def add_state(
    solver: z3.Solver,
    name_to_z3bool: Dict[str, z3.Bool],
    all_states: Set[State],
    state: State,
):
    if state not in all_states:
        all_states.add(state)
        name_to_z3bool.update(
            [
                (sample_to_name(state, a), z3.Bool(sample_to_name(state, a)))
                for a in ACT_SET
            ]
        )
        add_singleton_constraint(
            solver, [name_to_z3bool[sample_to_name(state, a)] for a in ACT_SET]
        )


def add_states_of_trace(
    solver: z3.Solver,
    name_to_z3bool: Dict[str, z3.Bool],
    all_states: Set[State],
    trace: Trace,
):
    for _, s, _, _, _ in trace.get_abstract_trace():
        add_state(solver, name_to_z3bool, all_states, s)
    add_state(solver, name_to_z3bool, all_states, trace[-1][4])


def add_decided_samples_condition(
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


def add_successful_condition(
    trace: Trace,
    all_states: Set[State],
    decided_samples: Dict[State, str],
    name_to_z3bool: Dict[str, z3.Bool],
    solver: z3.Solver,
):
    add_states_of_trace(solver, name_to_z3bool, all_states, trace)
    solver.add(
        z3.And(
            [
                name_to_z3bool[sample_to_name(s, a)]
                for _, s, a, _, _ in trace.get_abstract_trace()
            ]
        )
    )


def add_counterexample_condition(
    counterexample: Trace,
    all_states: Set[State],
    decided_samples: Dict[State, str],
    name_to_z3bool: Dict[str, z3.Bool],
    solver: z3.Solver,
):
    add_states_of_trace(solver, name_to_z3bool, all_states, counterexample)
    solver.add(
        z3.Or(
            [
                z3.Not(name_to_z3bool[sample_to_name(s, a)])
                for _, s, a, _, _ in counterexample.get_abstract_trace()
                if s not in decided_samples
            ]
        )
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


if __name__ == "__main__":
    args = get_arguments()
    verifier_rng = Random(args.verifier_seed)

    positive_demos = pickle_to_demo_traces(env_name=args.env_name)
    decided_samples = csv_to_positive_samples_dict(env_name=args.env_name)
    speculated_samples = dict()

    all_traces: List[Trace] = list()
    all_envs = set()
    working_envs = set()

    old_speculation_sets: List[Set[str]] = list()

    epoch = 0

    """ z3 related stuff """
    z3.set_option("smt.random_seed", args.z3_seed)
    name_to_z3bool = dict()
    all_states = set()
    solver = z3.Solver()

    for s in decided_samples:
        add_state(solver, name_to_z3bool, all_states, s)
    add_decided_samples_condition(decided_samples, name_to_z3bool, solver)

    while True:
        policy, _ = train_policy(
            env_name=args.env_name,
            decided_samples=decided_samples,
            speculated_samples=speculated_samples,
            seed=args.learner_seed,
            save=False,
        )
        if all_envs:
            psat, sat_trace_pairs = simulate_policy_on_list_of_envs(
                env_name=args.env_name,
                env_list=[e for e in all_envs],
                policy=policy,
                max_steps=args.max_steps,
            )
            working_envs = set()
            for e, (tsat, trace) in zip(all_envs, sat_trace_pairs):
                all_traces.append(trace)
                if tsat:
                    add_successful_condition(trace, all_states, decided_samples, name_to_z3bool, solver)
                else:
                    working_envs.add(e)
                    add_counterexample_condition(trace, all_states, decided_samples, name_to_z3bool, solver)
        debug(f"{len(working_envs) = } {len(all_envs) = } {len(all_traces) = }")

        if not working_envs:
            if epoch > 0:
                debug(f"End of CEGIS Epoch: {epoch}")
                debug(f"Number of Demo States: {len(decided_samples)}")
                num_new_states = len(all_states) - len(decided_samples)
                debug(f"Number of New States Seen: {num_new_states}")
                debug()
            epoch += 1
            debug(f"Start of CEGIS Epoch: {epoch}")
            miniround = 0
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

            all_traces.extend(traces)
            for counterexample in traces:
                all_envs.add(counterexample[0][0])
                working_envs.add(counterexample[0][0])
                add_counterexample_condition(
                    counterexample, all_states, decided_samples, name_to_z3bool, solver
                )
            debug(f"{len(working_envs) = } {len(all_envs) = } {len(all_traces) = }")

        if solver.check() != z3.sat:
            debug(f"Number of Demo States: {len(decided_samples)}")
            num_new_states = len(all_states) - len(decided_samples)
            debug(f"Number of New States Seen: {num_new_states}")
            raise Exception("UNSAT: Could not come up with a Speculated Sample set")

        assignment = solver.model()
        speculated_samples = dict()
        speculation_set = set()
        for s in all_states:
            for a in ACT_SET:
                if assignment[name_to_z3bool[sample_to_name(s, a)]]:
                    if s in decided_samples:
                        assert decided_samples[s] == a
                        continue
                    if s in speculated_samples:
                        raise Exception("Odd, 2 actions for same state")
                    speculated_samples[s] = a
                    speculation_set.add(sample_to_name(s, a))

        # for s in speculated_samples:
        #     debug(state_to_string(s, args.env_name), speculated_samples[s])
        # debug()
        debug(f"Completed {miniround = }")
        miniround += 1

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
