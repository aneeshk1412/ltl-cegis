import pickle
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

number_to_action_key = {
    0: 'left',
    2: 'up',
    3: 'pageup',
    5: ' '
}

def action_selection_policy_decision_tree(env: MiniGridEnv):
	with open('DT.model', 'rb') as f:
		model = pickle.load(f)
		state = extract_features(env)
		res = model.predict([state])
	return number_to_action_key[res]