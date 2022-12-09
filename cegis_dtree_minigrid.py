#!/usr/bin/env python3
import pickle
import gymnasium as gym
import pandas as pd
from sklearn import tree
from pprint import pprint
import matplotlib.pyplot as plt
import random

from minigrid.core.constants import ACT_KEY_TO_IDX

from verifier_minigrid import verify_action_selection_policy, verify_action_selection_policy_on_env
from demos_gen_minigrid import generate_demonstrations
from asp_minigrid import action_selection_policy_DoorKey_ground_truth, action_selection_policy_decision_tree
from dsl_minigrid import extract_features_DoorKey, feature_headers_DoorKey
from Minigrid.minigrid.minigrid_env import MiniGridEnv
from Minigrid.minigrid.wrappers import SymbolicObsWrapper
from Minigrid.minigrid.manual_control import ManualControl

def get_stem_and_loop(trace, demo_envs):
    hashes = [str(env) for env in demo_envs]
    for i, x in enumerate(hashes):
        try:
            idx = hashes[i+1:].index(x) + i+1
            return trace[:i], trace[i:idx]
        except ValueError:
            continue
    return trace, None


def train_and_save_model(dataset_dict, counter_dict, extra_dict=None, seed=None):
    if extra_dict is None:
        assert (set(dataset_dict.keys()) & set(counter_dict.keys()) == set([]))
        state_demos = pd.DataFrame(
            [state for state in dataset_dict] + [state for state in counter_dict])
        act_demos = pd.DataFrame([dataset_dict[state] for state in dataset_dict] + [
                                 counter_dict[state] for state in counter_dict])
    else:
        assert (set(dataset_dict.keys()) & set(counter_dict.keys()) == set([]))
        state_demos = pd.DataFrame([state for state in dataset_dict] + [
                                   state for state in counter_dict] + [state for state in extra_dict])
        act_demos = pd.DataFrame([dataset_dict[state] for state in dataset_dict] + [counter_dict[state]
                                 for state in counter_dict] + [extra_dict[state] for state in extra_dict])
    aspmodel = tree.DecisionTreeClassifier(
        class_weight='balanced', random_state=seed, max_features=None, max_leaf_nodes=None)
    clf = aspmodel.fit(state_demos, act_demos)
    with open('DT.model', 'wb') as f:
        pickle.dump(clf, f)
    return clf

def print_trace_stem_loop(trace, stem, loop):
    for line in trace:
        pprint(line)
    print()
    print("STEM: ")
    for line in stem:
        pprint(line)
    print()
    print("LOOP: ")
    if loop is not None:
        for line in loop:
            pprint(line)
        print()


def random_sampling_algorithm(dataset_dict, trace):
    """ Sample new action randomly for the Last (state, action) pair to correct it (as long as it is consistent with demonstration) """
    for x, y in reversed(trace):
        if x in dataset_dict:
            continue
        # Make a more informed decision
        z = random.sample(list(set(ACT_KEY_TO_IDX.keys()) - set([y])), 1)
        return x, z[0]
    return None, None


def random_loop_correction(dataset_dict, trace, demo_envs):
    """ If a loop is present, correct the first (state, action) pair using a new random action (as long as it is consistent with demonstration) """
    stem, loop = get_stem_and_loop(trace, demo_envs)
    if loop is None:
        return random_sampling_algorithm(dataset_dict, stem)
    for x, y in loop:
        if x in dataset_dict:
            continue
        z = random.sample(list(set(ACT_KEY_TO_IDX.keys()) - set([y])), 1)
        return x, z[0]
    return None, None

# def loop_correction_with_one_step_loop_check(dataset_dict, trace, demo_envs, counter_dict):
#     stem, loop = get_stem_and_loop(trace, demo_envs)
#     if loop is None:
#         for (x, y), env in reversed(zip(stem, demo_envs)):
#             if x in dataset_dict:
#                 continue
#             for act in ACT_KEY_TO_IDX.keys():
#                 if act != y:
#                     aspmodel =
#                     sat, _, _ = verify_action_selection_policy_on_env(env, )
#                     if sat:
#                         return x, y
#         return None, None
#     for x, y in loop:
#         if x in dataset_dict:
#             continue
#         z = random.sample(list(set(ACT_KEY_TO_IDX.keys()) - set([y])), 1)
#         return x, z[0]
#     return None, None

def policy_apply_model(env: MiniGridEnv):
    with open('DT.model', 'rb') as f:
        model = pickle.load(f)
        state = extract_features_DoorKey(env)
        res = (model.predict([state]))
    return res[0]

if __name__ == "__main__":
    gen_seed = None
    env_name = "MiniGrid-DoorKey-16x16-v0"
    _RUN_SIMULATION = False # set this to true to test the llearned model through thr cegis loop. Set it to false to run the cegis loop
    if _RUN_SIMULATION:
        env: MiniGridEnv = gym.make(env_name, tile_size=42)
        env = SymbolicObsWrapper(env)
        manual_control = ManualControl(
            env, agent_view=False, seed=gen_seed, policy=policy_apply_model, features=feature_headers_DoorKey())
        manual_control.start()
    else:
        plot_tree = True
        verifier_seed = None

        dtree_seed = None

        demos = generate_demonstrations(
            env_name, action_selection_policy_DoorKey_ground_truth, seed=gen_seed, num_demos=1, timeout=100)
        dataset_dict = dict(demos)
        counter_dict = dict()

        sat = False
        epoch = 0
        while not sat:
            print(f"{epoch = }")
            aspmodel = train_and_save_model(
                dataset_dict, counter_dict, seed=dtree_seed)

            def action_selection_policy(env): return action_selection_policy_decision_tree(
                env, aspmodel, extract_features_DoorKey)
            sat, trace, demo_envs = verify_action_selection_policy(
                env_name, action_selection_policy, seed=verifier_seed, num_trials=15, timeout=100)

            print(f"{sat = }")
            if not sat:
                x, z = random_loop_correction(dataset_dict, trace, demo_envs)
                counter_dict[x] = z
                print(f"Added Demonstration: {x} -> {z}")
            print()
            epoch += 1

        if plot_tree:
            tree.plot_tree(aspmodel, max_depth=None, class_names=sorted(set(dataset_dict.values()) | set(counter_dict.values(
            ))), label='none', precision=1, feature_names=feature_headers_DoorKey(), rounded=True, fontsize=5, proportion=True)
            plt.show()
