#!/usr/bin/env python3

import gymnasium as gym

from learner_minigrid import learn
from dsl_minigrid import feature_mapping
from simulator_minigrid import simulate_policy_on_state
from commons_minigrid import Decisions, pickle_to_demo_traces, PartialGraph, parse_args, debug, get_decisions

from minigrid.minigrid_env import MiniGridEnv


if __name__ == "__main__":
    args = parse_args()

    feature_fn = feature_mapping[args.env_name]
    partialMDP = PartialGraph(feature_fn)

    demonstrations = pickle_to_demo_traces(args.env_name)
    for demo in demonstrations:
        partialMDP.add_trace(demo)

    decisions = Decisions()
    for demo in demonstrations:
        decisions.add_decision_list([(feature_fn(s), a) for s, a, _ in demo])

    policy, model = learn(decisions)

    randomstate: MiniGridEnv = gym.make(args.env_name, tile_size=32)
    cegis_epochs = 0
    count = 0
    while True:
        randomstate.reset()
        sat, trace = simulate_policy_on_state(
            state=randomstate,
            policy=policy,
            feature_fn=feature_fn,
            spec=args.spec,
            args=args,
        )
        count += 1

        if count >= args.threshold:
            print("Found Satisfying Policy")
            break
        partialMDP.add_trace(trace)
        if not sat:
            print(f"CEGIS Epoch: {cegis_epochs}, Counterexample found after : {count} runs")
            cegis_epochs += 1
            count = 0
            decisions = get_decisions(partialMDP, args.spec)
            policy, model = learn(decisions)
            # Assert success on all known environments?
