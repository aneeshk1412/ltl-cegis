#!/usr/bin/env python3

import gymnasium as gym

from learner_minigrid import learn
from dsl_minigrid import feature_mapping
from simulator_minigrid import simulate_policy_on_state
from commons_minigrid import (
    Decisions,
    pickle_to_demo_traces,
    AbstractGraph,
    parse_args,
    get_decisions,
)

from minigrid.minigrid_env import MiniGridEnv


if __name__ == "__main__":
    args = parse_args()

    feature_fn = feature_mapping[args.env_name]
    abstract_graph = AbstractGraph()

    demonstrations = pickle_to_demo_traces(args.env_name)
    for demo in demonstrations:
        for s, a, s_p in demo:
            abstract_graph.add_edge(feature_fn(s), feature_fn(s_p), a)

    decisions = Decisions()
    for demo in demonstrations:
        decisions.add_decision_list([(feature_fn(s), a) for s, a, _ in demo])


    randomstate: MiniGridEnv = gym.make(args.env_name, tile_size=32)
    cegis_epochs = 0
    while True:
        policy, model = learn(decisions, args=args)
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
            for s, a, s_p in trace:
                abstract_graph.add_edge(feature_fn(s), feature_fn(s_p), a)
            count += 1
            print(f"Number of Checks: {count}", end="\r")
            if not sat:
                break
            if count >= args.threshold:
                print("Found Satisfying Policy")
                break
        if count >= args.threshold:
            break
        cegis_epochs += 1
        print(f"CEGIS Epoch: {cegis_epochs}, Counterexample found at Run: {count}")
        decisions = get_decisions(abstract_graph, args.spec)
        policy, model = learn(decisions)
        # Assert success on all known environments?
    abstract_graph.show_graph(f"iter{cegis_epochs}.html")
