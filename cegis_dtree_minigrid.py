#!/usr/bin/env python3
import pandas as pd
from sklearn import tree
from pprint import pprint
import matplotlib.pyplot as plt
import random

from minigrid.core.constants import ACT_KEY_TO_IDX

from verifier_minigrid import verify_action_selection_policy
from demos_gen_minigrid import generate_demonstrations
from asp_minigrid import action_selection_policy_DoorKey_ground_truth, action_selection_policy_decision_tree
from dsl_minigrid import extract_features_DoorKey, feature_headers_DoorKey

def get_stem_and_loop(trace, demo_envs):
    hashes = [str(env) for env in demo_envs]
    for i, x in enumerate(hashes):
        try:
            idx = hashes[i+1:].index(x) + i+1
            return trace[:i], trace[i:idx]
        except ValueError:
            continue
    return trace, None

def train_model(dataset_dict, counter_dict, seed=None):
    assert(set(dataset_dict.keys()) & set(counter_dict.keys()) == set([]))
    state_demos = pd.DataFrame([state for state in dataset_dict] + [state for state in counter_dict]).astype(int)
    act_demos = pd.DataFrame([dataset_dict[state] for state in dataset_dict] + [counter_dict[state] for state in counter_dict])
    aspmodel = tree.DecisionTreeClassifier(class_weight='balanced', random_state=seed, max_features=None, max_leaf_nodes=None)
    return aspmodel.fit(state_demos, act_demos)

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

plot_tree = False
verifier_seed = None
gen_seed = 300
dtree_seed = 300

demos = generate_demonstrations("MiniGrid-DoorKey-16x16-v0", action_selection_policy_DoorKey_ground_truth, seed=gen_seed, num_demos=2, timeout=100)
dataset_dict = dict(demos)
counter_dict = dict()

sat = False
epoch = 0
while not sat:
    print(f"{epoch = }")
    aspmodel = train_model(dataset_dict, counter_dict, seed=dtree_seed)
    action_selection_policy = lambda env: action_selection_policy_decision_tree(env, aspmodel, extract_features_DoorKey)
    if plot_tree:
        tree.plot_tree(aspmodel, max_depth=None, class_names=sorted(set(dataset_dict.values())), label='none', precision=1, feature_names=feature_headers_DoorKey(), rounded=True, fontsize=5, proportion=True)
        plt.show()
    sat, trace, demo_envs = verify_action_selection_policy("MiniGrid-DoorKey-16x16-v0", action_selection_policy, seed=verifier_seed, num_trials=100, timeout=100)

    print(f"{sat = }")
    if not sat:
        for x, y in reversed(trace):
            if x in dataset_dict:
                continue
            z = random.sample(list(set(ACT_KEY_TO_IDX.keys()) - set([y])), 1)
            counter_dict[x] = z[0]
            break
    print()
    epoch += 1

# stem, loop = get_stem_and_loop(trace, demo_envs)
#         print_trace_stem_loop(trace, stem, loop)
#         if loop is not None:
#             for x, y in loop:
#                 if x in dataset_dict:
#                     continue
#                 counter_dict[x] = z
#                 break
#         else:
#             for x, y in reversed(stem):