#!/usr/bin/env python3

import z3
import gymnasium as gym
from copy import deepcopy
from random import Random
from collections import deque
from typing import List, Tuple

from learner_minigrid import learn
from dsl_minigrid import feature_mapping
from simulator_minigrid import simulate_policy_on_state, step
from commons_minigrid import (
    debug,
    Trace,
    Decisions,
    parse_args,
    ConcreteGraph,
    AbstractGraph,
    pickle_to_demo_traces,
    get_decisions_reachability,
)

from minigrid.minigrid_env import MiniGridEnv


if __name__ == "__main__":
    args = parse_args()
    randomstate_seed = Random(args.simulator_seed)

    feature_fn = feature_mapping[args.env_name]
    graph = AbstractGraph()

    """ Get initial Decisions from Demonstrations """
    demonstrations = pickle_to_demo_traces(args.env_name)
    decisions = Decisions()
    for demo in demonstrations:
        decisions.add_decision_list([(feature_fn(s), a) for s, a, _ in demo])

    """ Initialize Abstract Graph """
    for demo in demonstrations:
        graph.add_trace(demo, feature_fn)
        graph.set_reachable(feature_fn(demo[-1][2]))

    randomstate: MiniGridEnv = gym.make(args.env_name, tile_size=32)
    cegis_epochs = 0

    while True:
        policy, model = learn(decisions, args=args)
        count = 0
        traces: List[Tuple[bool, Trace]] = []

        while True:
            randomstate.reset(seed=randomstate_seed.randint(0, 1e10))
            sat, trace = simulate_policy_on_state(
                state=randomstate,
                policy=policy,
                feature_fn=feature_fn,
                spec=args.spec,
                args=args,
                show_if_unsat=False,
            )
            count += 1
            traces.append((sat, trace))

            debug(f"Number of Checks: {count}", end="\r")

            if not sat:
                break
            if count >= args.threshold:
                print(f"Found Satisfying Policy. Threshold {count} runs")
                break

        if count >= args.threshold:
            break

        cegis_epochs += 1
        print(f"CEGIS Epoch: {cegis_epochs}, Counterexample found at Run: {count}")

        for sat, tau in traces:
            graph.add_trace(tau, feature_fn)
            if sat:
                graph.set_reachable(feature_fn(tau[-1][2]))

        counterex = trace

        state_queue = deque([deepcopy(s) for s in reversed(counterex.get_state_sequence())])
        is_trace_reaching = False
        while state_queue:
            current_s = state_queue.pop()
            if graph.can_reach(feature_fn(current_s)):
                is_trace_reaching = True
                break
            untried_acts = deepcopy(graph.get_untried_acts(feature_fn(current_s)))
            for a in untried_acts:
                current_s, a, next_s, reward, terminated, _ = step(current_s, a)
                graph.add_transition((current_s, a, next_s), feature_fn)
                if terminated and reward > 0.0:
                    graph.set_reachable(feature_fn(next_s))
                if graph.can_reach(feature_fn(next_s)):
                    is_trace_reaching = True
                    break
                state_queue.append(next_s)
            if is_trace_reaching:
                break

        if not is_trace_reaching:
            raise Exception("What!")
        decisions = get_decisions_reachability(graph, feature_fn)
