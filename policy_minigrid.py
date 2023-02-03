import pandas as pd
from minigrid.minigrid_env import MiniGridEnv

from dsl_minigrid import *


def policy_DoorKey_ground_truth(env: MiniGridEnv):
    if is_present(env, "key") and check(env, env.front_pos, "key"):
        return "pageup"
    if is_present(env, "key") and is_agent_facing(env, get_nearest(env, "key")):
        return "up"
    if is_present(env, "key") and is_at_agents_right(env, get_nearest(env, "key")):
        return 'right'
    if is_present(env, "key") and is_at_agents_left(env, get_nearest(env, "key")):
        return 'left'

    if is_present(env, "door") and check(env, env.front_pos, "door"):
        return " "
    if is_present(env, "door") and is_agent_facing(env, get_nearest(env, "door")) and not check(env, env.front_pos, "wall"):
        return "up"
    if is_present(env, "door") and is_at_agents_right(env, get_nearest(env, "door")):
        return 'right'
    if is_present(env, "door") and is_at_agents_left(env, get_nearest(env, "door")):
        return 'left'

    if is_present(env, "goal") and is_agent_facing(env, get_nearest(env, "goal")):
        return "up"
    if is_present(env, "goal") and is_at_agents_right(env, get_nearest(env, "goal")):
        return 'right'
    if is_present(env, "goal") and is_at_agents_left(env, get_nearest(env, "goal")):
        return 'left'

    return "up"


def policy_DoorKey_wrong(env: MiniGridEnv):
    if is_present(env, "door") and check(env, env.front_pos, "door"):
        return " "
    if is_present(env, "door") and is_agent_facing(env, get_nearest(env, "door")) and not check(env, env.front_pos, "wall"):
        return "up"
    if is_present(env, "door"):
        return "left"
    if is_present(env, "key") and check(env, env.front_pos, "key"):
        return "pageup"
    if is_present(env, "key") and is_agent_facing(env, get_nearest(env, "key")):
        return "up"
    if is_present(env, "key"):
        return "left"
    return "up"


def policy_decision_tree(env: MiniGridEnv, model, extract_features):
    state = pd.DataFrame([extract_features(env)])
    return model.predict(state)[0]


ground_truth_asp_register = {
    "MiniGrid-DoorKey-16x16-v0": policy_DoorKey_ground_truth,
}
