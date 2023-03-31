#!/usr/bin/env python3

import gymnasium as gym
from random import Random
from copy import deepcopy

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import ACT_STR_TO_ENUM


def step(
    env: MiniGridEnv,
    policy,
    path_features,
    feature_fn,
    prev_env_id_set,
):
    a = policy(feature_fn(env))
    action = ACT_STR_TO_ENUM[a]
    s = deepcopy(env)
    _, reward, terminated, truncated, _ = env.step(action)
    s_p = deepcopy(env)
    path_features.add(feature_fn(env))
    done = None
    if truncated or env.identifier() in prev_env_id_set:
        done = False
    elif terminated and reward < 0.0:
        done = False
    elif terminated:
        done = True
    return done, (s, a, s_p)


def simulate_policy_on_env(
    env: MiniGridEnv,
    policy,
    seed,
    max_steps,
    feature_fn,
    specification,
):
    trace = list()
    prev_env_id_set = set()
    path_features = PathFeatures()
    environ = deepcopy(env)
    environ.reset(soft=True, seed=seed, max_steps=max_steps)
    while True:
        prev_env_id_set.add(environ.identifier())
        done, transition = step(
            env=environ,
            policy=policy,
            path_features=path_features,
            feature_fn=feature_fn,
            specification=specification,
            prev_env_id_set=prev_env_id_set,
        )
        trace.append(transition)
        if done:
            break
    sat = path_features.satisfies(specification)
    return sat, trace
