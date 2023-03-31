#!/usr/bin/env python3

from copy import deepcopy
from typing import Set, Tuple

from custom_types_minigrid import (
    State,
    Policy,
    Feature_Func,
    Transition,
    Specification,
    Trace,
)

from minigrid.core.constants import ACT_STR_TO_ENUM


def step(
    state: State,
    policy: Policy,
    feature_fn: Feature_Func,
    prev_env_id_set: Set[int],
) -> Tuple[bool, Transition]:
    a = policy(feature_fn(state))
    action = ACT_STR_TO_ENUM[a]
    s = deepcopy(state)  ## Add to indexer
    _, reward, terminated, truncated, _ = state.step(action=action)
    s_p = deepcopy(state)  ## Add to indexer
    done = None
    if truncated or state.identifier() in prev_env_id_set:
        done = True
    elif terminated and reward < 0.0:
        done = True
    elif terminated:
        done = True
    ## return done, (idx, a, idx_p)
    return done, (s, a, s_p)


def simulate_policy_on_state(
    state: State,
    policy: Policy,
    feature_fn: Feature_Func,
    spec: Specification,
    seed: int | None = None,
    max_steps: int = 100,
):
    trace = list()
    prev_env_id_set = set()
    env = deepcopy(state)
    env.reset(soft=True, seed=seed, max_steps=max_steps)
    while True:
        prev_env_id_set.add(env.identifier())
        done, transition = step(
            state=env,
            policy=policy,
            feature_fn=feature_fn,
            prev_env_id_set=prev_env_id_set,
        )
        trace.append(transition)
        if done:
            break
    ## Convert from IndexTransition to Transition
    trace = Trace(trace)
    sat = trace.satisfies(spec=spec, feature_fn=feature_fn)
    return sat, trace


if __name__ == "__main__":
    import argparse
    import gymnasium as gym

    from custom_types_minigrid import Action

    from minigrid.minigrid_env import MiniGridEnv

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
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    args = parser.parse_args()

    state: MiniGridEnv = gym.make(args.env, tile_size=args.tile_size)
    state.reset()
    policy = lambda feats: Action("forward")

    sat, trace = simulate_policy_on_state(
        state=state,
        policy=policy,
        feature_fn=lambda state: state,
        spec="",
        seed=args.seed,
        max_steps=args.max_steps,
    )
    for s, a, s_p in trace:
        print(s)
        print(a)
    print(s_p)
    # assert trace[0][0].identifier() == trace[-1][2].identifier()
