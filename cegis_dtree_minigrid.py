#!/usr/bin/env python3
import pandas as pd
from sklearn import tree
from pprint import pprint
import matplotlib.pyplot as plt

from verifier_minigrid import verify_action_selection_policy
from demos_gen_minigrid import generate_demonstrations
from asp_minigrid import action_selection_policy_DoorKey_ground_truth, action_selection_policy_decision_tree
from dsl_minigrid import extract_features_DoorKey, feature_headers_DoorKey

plot_tree = True
seed = None

demos = generate_demonstrations("MiniGrid-DoorKey-16x16-v0", action_selection_policy_DoorKey_ground_truth, seed=seed, num_demos=5, timeout=100)
state_demos = pd.DataFrame([d[0] for d in demos]).astype(int)
act_demos = pd.DataFrame([d[1] for d in demos])

aspmodel = tree.DecisionTreeClassifier(class_weight='balanced', random_state=seed, max_features=None, max_leaf_nodes=None)
aspmodel = aspmodel.fit(state_demos, act_demos)
action_selection_policy = action_selection_policy_decision_tree

if plot_tree:
    tree.plot_tree(aspmodel, max_depth=None, class_names=sorted(act_demos[0].unique()), label='none', precision=1, feature_names=feature_headers_DoorKey(), rounded=True, fontsize=5, proportion=True)
    plt.show()

sat = False
epoch = 0
while not sat:
    print(f"{epoch = }")
    action_selection_policy = lambda env: action_selection_policy_decision_tree(env, aspmodel, extract_features_DoorKey)
    sat, trace, demo_envs = verify_action_selection_policy("MiniGrid-DoorKey-16x16-v0", action_selection_policy, seed=seed, timeout=100)

    print(f"{sat = }")
    if not sat:
        for line in trace:
            pprint(line)
        pass
    print()
    epoch += 1
