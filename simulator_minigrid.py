#!/usr/bin/env python3

from copy import deepcopy
from typing import Set, Tuple

from commons_minigrid import (
    State,
    Policy,
    Feature_Func,
    Transition,
    Specification,
    Trace,
    satisfies,
    Arguments,
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
    args: Arguments,
):
    trace = list()
    prev_env_id_set = set()
    env = deepcopy(state)
    env.reset(soft=True, seed=args.simulator_seed, max_steps=args.max_steps)
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
    ## Convert from IndexTransition to Transition before sending
    trace = Trace(trace)
    sat = satisfies(trace=trace, spec=spec, feature_fn=feature_fn)
    return sat, trace


if __name__ == "__main__":
    import gymnasium as gym

    from commons_minigrid import Action, parse_args
    from dsl_minigrid import feature_mapping

    from minigrid.minigrid_env import MiniGridEnv

    args = parse_args()

    # policy = lambda feats: Action("forward")
    policy = (
        lambda feats: Action("right")
        if feats["check_agent_front_pos__wall"]
        else Action("forward")
    )

    state: MiniGridEnv = gym.make(args.env_name, tile_size=args.tile_size)
    state.reset()

    sat, trace = simulate_policy_on_state(
        state=state,
        policy=policy,
        feature_fn=feature_mapping[args.env_name],
        spec=args.spec,
        args=args,
    )
    print(f"{sat = }")
