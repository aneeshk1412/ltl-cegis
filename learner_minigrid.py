#!/usr/bin/env python3

from typing import Set
import pickle
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

from dsl_minigrid import header_register


def train_policy(
    decided_samples: dict,
    speculated_samples: dict,
    seed: int | None = None,
    save: bool = False,
):
    assert set(decided_samples.keys()) & set(speculated_samples.keys()) == set()
    states = pd.DataFrame(
        [s for s in decided_samples] + [s for s in speculated_samples]
    )
    actions = pd.DataFrame(
        [decided_samples[s] for s in decided_samples]
        + [speculated_samples[s] for s in speculated_samples]
    )
    model = tree.DecisionTreeClassifier(
        class_weight="balanced",
        random_state=seed,
        max_features=None,
        max_leaf_nodes=None,
    )
    policy = model.fit(states, actions)
    if save:
        with open("model.pkl", "wb") as f:
            pickle.dump(policy, f)
    return policy


def plot_policy(
    policy: tree.DecisionTreeClassifier, class_names: Set[str], env_name: str
):
    tree.plot_tree(
        policy,
        max_depth=None,
        class_names=class_names,
        label="none",
        precision=1,
        feature_names=header_register[env_name],
        rounded=True,
        fontsize=5,
        proportion=True,
    )
    plt.show()
