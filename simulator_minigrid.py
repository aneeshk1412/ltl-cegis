#!/usr/bin/env python3

from copy import deepcopy
from typing import Set, Tuple, List

from commons_minigrid import (
    State,
    Action,
    Policy,
    Feature_Func,
    Transition,
    Specification,
    Trace,
    Arguments,
)

from minigrid.core.constants import ACT_STR_TO_ENUM
from minigrid.utils.window import Window


def step(state: State, act: Action):
    """Given a State and an Action, sample the next State.
    The given State object gets modified to the next State.
    """
    action = ACT_STR_TO_ENUM[act]
    s = deepcopy(state)
    _, reward, terminated, truncated, _ = state.step(action=action)
    s_p = deepcopy(state)
    ## To handle a bug where if reaches the goal without picking all keys,
    ## the state is satisfying, just use the old state (a self loop)
    if terminated and reward < 0.0:
        s_p = deepcopy(s)
    return s, act, s_p, reward, terminated, truncated


def step_wrapper(
    state: State,
    policy: Policy,
    feature_fn: Feature_Func,
    prev_env_id_set: Set[int],
) -> Tuple[bool, Transition]:
    """Takes a transition from the given State using the
    given Policy and Feature Function. Stop early if a cycle
    is detected (requires that state has an identifier function).
    """
    a = policy(feature_fn(state))
    s, a, s_p, reward, terminated, truncated = step(state, a)
    done, sat = None, None
    if truncated or state.identifier() in prev_env_id_set:
        done, sat = True, False
    elif terminated and reward < 0.0:
        done, sat = True, False
    elif terminated:
        done, sat = True, True
    return done, sat, (s, a, s_p)


def intersperse(lst, item, n):
    res = []
    for i in range(len(lst)):
        res.append(lst[i])
        if i % n == n - 1:
            res.append(item)
    return res


def show_policy_on_window(
    state: State, policy: Policy, feature_fn: Feature_Func, args: Arguments
):
    """Shows the running of a Policy starting from a given State on a Window."""
    env = deepcopy(state)
    env.reset(soft=True, seed=args.simulator_seed, max_steps=args.max_steps)
    window = Window("minigrid - " + str(env.__class__))
    frame = env.get_frame(agent_pov=False)
    window.show_img(frame)
    while True:
        act = policy(feature_fn(state))
        action = ACT_STR_TO_ENUM[act]
        _, reward, terminated, truncated, _ = env.step(action=action)
        frame = env.get_frame(agent_pov=False)
        window.show_img(frame)
        feats = "    ".join(
            intersperse([k for k, v in feature_fn(state).items() if v], "\n", 3)
        )
        window.set_caption(f"Action: {act}\n Features: {feats}")
        if truncated:
            break
        elif terminated and reward < 0.0:
            break
        elif terminated:
            break
    window.close()


def simulate_policy_on_state(
    state: State,
    policy: Policy,
    feature_fn: Feature_Func,
    spec: Specification,
    args: Arguments,
    show_if_unsat: bool = False,
):
    """Simulate a Policy starting from a State
    and check if it satisfies a Specification.
    Currently the Specification is encoded in the task environment.
    """
    trace = list()
    prev_env_id_set = set()
    env = deepcopy(state)
    env.reset(soft=True, seed=args.simulator_seed, max_steps=args.max_steps)
    while True:
        prev_env_id_set.add(env.identifier())
        done, sat, transition = step_wrapper(
            state=env,
            policy=policy,
            feature_fn=feature_fn,
            prev_env_id_set=prev_env_id_set,
        )
        trace.append(transition)
        if done:
            break
    if not sat and (show_if_unsat or args.show_if_unsat):
        show_policy_on_window(
            state=state, policy=policy, feature_fn=feature_fn, args=args
        )
    trace = Trace(trace)
    # assert sat == satisfies(trace, spec, feature_fn)
    return sat, trace


def simulate_policy_on_list_of_states(
    state_list: List[State],
    policy: Policy,
    feature_fn: Feature_Func,
    spec: Specification,
    args: Arguments,
):
    sat_trace_pairs = [
        simulate_policy_on_state(
            state=s, policy=policy, feature_fn=feature_fn, spec=spec, args=args
        )
        for s in state_list
    ]
    all_sat = all(sat for sat, _ in sat_trace_pairs)
    return all_sat, sat_trace_pairs


if __name__ == "__main__":
    import gymnasium as gym

    from commons_minigrid import Action, parse_args
    from dsl_minigrid import feature_mapping

    from minigrid.minigrid_env import MiniGridEnv

    args = parse_args()

    policy = lambda feats: Action("forward")
    # policy = (
    #     lambda feats: Action("right")
    #     if feats["check_agent_front_pos__wall"]
    #     else Action("forward")
    # )

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
