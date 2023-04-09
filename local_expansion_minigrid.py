#!/usr/bin/env python3

import gymnasium as gym

from learner_minigrid import learn, augment_policy
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
from minigrid.core.constants import ACT_SET


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
        traces = []
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
            traces.append(trace)

            print(f"Number of Checks: {count}", end="\r")
            if not sat:
                break
            if count >= args.threshold:
                print(f"Found Satisfying Policy. Threshold {count} runs")
                break

        if count >= args.threshold:
            break

        cegis_epochs += 1
        print(f"CEGIS Epoch: {cegis_epochs}, Counterexample found at Run: {count}")

        for tau in traces:
            for s, a, s_p in tau:
                abstract_graph.add_edge(feature_fn(s), feature_fn(s_p), a)

        ## Expanding the graph locally around the trace
        for s, a, _ in trace:
            for a_new in ACT_SET - {a}:
                local_decision_change = Decisions()
                local_decision_change.add_decision(feature_fn(s), a_new)
                aug_policy = augment_policy(policy, local_decision_change)
                new_sat, new_trace = simulate_policy_on_state(
                    state=s,
                    policy=aug_policy,
                    feature_fn=feature_fn,
                    spec=args.spec,
                    args=args,
                )
                for s, a, s_p in new_trace:
                    abstract_graph.add_edge(feature_fn(s), feature_fn(s_p), a)

        abstract_graph.show_graph(f"graph.html")
        decisions = get_decisions(abstract_graph, args.spec)
        # Assert success on all known environments?
