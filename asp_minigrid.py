import pandas as pd
from minigrid.minigrid_env import MiniGridEnv

from dsl_minigrid import *


def action_selection_policy_DoorKey_ground_truth(env: MiniGridEnv):
    if is_present(env, "key") and check(env, env.front_pos, "key"):
        return "pageup"
    if is_present(env, "key") and is_agent_facing(env, get_nearest(env, "key")):
        return "up"
    if is_present(env, "key"):
        return "left"
    if is_present(env, "door") and check(env, env.front_pos, "door"):
        return " "
    if is_present(env, "door") and is_agent_facing(env, get_nearest(env, "door")) and not check(env, env.front_pos, "wall"):
        return "up"
    if is_present(env, "door"):
        return "left"
    if is_present(env, "goal") and is_agent_facing(env, get_nearest(env, "goal")):
        return "up"
    if is_present(env, "goal"):
        return "left"
    return "up"

# another version of the ground truth where 'right' and 'left' actions are BOTH used
def action_selection_policy_DoorKey_ground_truth_right(env: MiniGridEnv):
    if is_present(env, "key") and check(env, env.front_pos, "key"):
        return "pageup"
    if is_present(env, "key") and is_agent_facing(env, get_nearest(env, "key")):
        return "up"
    if is_present(env, "key"):
        if is_at_agents_right(env, get_nearest(env, "key")):
            return 'right'
        else:
            return 'left'
    if is_present(env, "door") and check(env, env.front_pos, "door"):
        return " "
    if is_present(env, "door") and is_agent_facing(env, get_nearest(env, "door")) and not check(env, env.front_pos, "wall"):
        return "up"
    if is_present(env, "door"):
        if is_at_agents_right(env, get_nearest(env, "door")):
            return 'right'
        else:
            return 'left'
    if is_present(env, "goal") and is_agent_facing(env, get_nearest(env, "goal")):
        return "up"
    if is_present(env, "goal"):
        if is_at_agents_right(env, get_nearest(env, "goal")):
            return 'right'
        else:
            return 'left'
    return "up"


def action_selection_policy_DoorKey_wrong(env: MiniGridEnv):
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


def action_selection_policy_decision_tree(env: MiniGridEnv, model, extract_features):
    state = pd.DataFrame([extract_features(env)])
    action = model.predict(state)[0]
    return action


ground_truth_asp_register = {
    "MiniGrid-DoorKey-16x16-v0": action_selection_policy_DoorKey_ground_truth_right,
}
