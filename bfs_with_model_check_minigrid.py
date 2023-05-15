#!/usr/bin/env python3

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
    AbstractGraph,
    pickle_to_demo_traces,
    get_decisions_reachability,
    get_decisions
)

from minigrid.minigrid_env import MiniGridEnv


if __name__ == "__main__":
    args = parse_args()
    randomstate_seed = Random(args.simulator_seed)

    feature_fn = feature_mapping[args.env_name]
    abstract_graph = AbstractGraph()

    """ Get initial Decisions from Demonstrations """
    demonstrations = pickle_to_demo_traces(args.env_name)
    decisions = Decisions()
    for demo in demonstrations:
        decisions.add_decision_list([(feature_fn(s), a) for s, a, _ in demo])

    """ Initialize Abstract Graph """
    for demo in demonstrations:
        for u, a, s_p in demo:
            abstract_graph.add_edge(
                feature_fn(u), feature_fn(s_p), a, u.identifier() == s_p.identifier()
            )
        abstract_graph.set_reachable(feature_fn(s_p))

    target_feats = feature_fn(s_p)

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
                show_if_unsat=True,
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
            for u, a, s_p in tau:
                abstract_graph.add_edge(
                    feature_fn(u),
                    feature_fn(s_p),
                    a,
                    u.identifier() == s_p.identifier(),
                )
            if sat:
                abstract_graph.set_reachable(feature_fn(s_p))

        counterex = trace

        # if all(abstract_graph.can_reach(feature_fn(s)) for s, _, _ in counterex):
        #     for s, a, _ in counterex:
        #         print(abstract_graph.feats_to_ids[feature_fn(s)], abstract_graph.can_reach(feature_fn(s)), a)
        #     abstract_graph.show_graph(f"oddgraph.html")
        #     sys = get_system(abstract_graph, with_dummy=False)
        #     with open("oddgraph.prism", "w") as f:
        #         f.writelines(sys)
        #     # exit()
        # pprint(abstract_graph.reaching.reachable)

        state_queue = deque(
            [deepcopy(s) for s, _, _ in reversed(counterex)] + [counterex[-1][2]]
        )
        is_trace_reaching = False
        while state_queue:
            current_s = state_queue.pop()
            current_feats = feature_fn(current_s)
            current_feats_idx = abstract_graph.get_index(current_feats)

            if abstract_graph.can_reach(current_feats):
                is_trace_reaching = True
                break
            untried_acts = deepcopy(
                abstract_graph.ids_to_untried_acts[current_feats_idx]
            )
            for a in untried_acts:
                current_s, _, next_s, reward, terminated, _ = step(current_s, a)
                next_feats = feature_fn(next_s)
                next_feats_idx = abstract_graph.get_index(next_feats)

                abstract_graph.add_edge(
                    current_feats,
                    next_feats,
                    a,
                    current_s.identifier() == next_s.identifier(),
                )

                if terminated and reward > 0.0:
                    abstract_graph.set_reachable(next_feats)

                if abstract_graph.can_reach(next_feats):
                    is_trace_reaching = True
                    break
                state_queue.append(next_s)
            if is_trace_reaching:
                break

        abstract_graph.show_graph(f"graph.html")
        if not is_trace_reaching:
            raise Exception("What!")

        decisions = get_decisions(abstract_graph, args.spec)
        # decisions = get_decisions_reachability(abstract_graph, target_feats)
