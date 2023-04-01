#!/usr/bin/env python3

import argparse
import gymnasium as gym

from learner_minigrid import learn
from dsl_minigrid import features_empty
from simulator_minigrid import simulate_policy_on_state
from commons_minigrid import Decisions, pickle_to_demo_traces, PartialMDP

from minigrid.minigrid_env import MiniGridEnv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-MultiRoom-N6-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--max-steps", type=int, help="number of steps to timeout after", default=100
    )
    parser.add_argument(
        "--threshold", type=int, help="threshold to declare safe", default=200
    )
    args = parser.parse_args()

    partialMDP = PartialMDP()

    demonstrations = pickle_to_demo_traces(args.env)
    for demo in demonstrations:
        partialMDP.add_trace(demo)

    decisions = Decisions()
    for demo in demonstrations:
        decisions.add_decision_list([(features_empty(s), a) for s, a, _ in demo])

    policy, model = learn(decisions)

    randomstate: MiniGridEnv = gym.make(args.env, tile_size=32)
    count = 0
    while True:
        randomstate.reset()
        sat, trace = simulate_policy_on_state(
            state=randomstate,
            policy=policy,
            feature_fn=features_empty,
            spec='P>=1 [F "is_agent_on__goal"]',
        )
        count += 1
        if count >= args.threshold:
            print("Found Satisfying Policy")
            break
        partialMDP.add_trace(trace)
        if not sat:
            print(f"Counterexample found after : {count} runs")
            count = 0
            # decisions = partialMDP.get_decisions(spec='P>=1 [F "is_agent_on__goal"]', feature_fn=features_empty)
            policy, model = learn(decisions)
