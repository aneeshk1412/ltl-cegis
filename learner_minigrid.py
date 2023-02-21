#!/usr/bin/env python3

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import Set, Tuple, Callable
from sklearn.tree import DecisionTreeClassifier, plot_tree

from policy_minigrid import policy_decision_tree
from dsl_minigrid import header_register, feature_register

from minigrid.minigrid_env import MiniGridEnv


def train_policy(
    env_name: str,
    decided_samples: dict,
    speculated_samples: dict,
    seed: int | None = None,
    save: bool = False,
) -> Tuple[Callable[[MiniGridEnv], str], DecisionTreeClassifier]:
    assert set(decided_samples.keys()) & set(speculated_samples.keys()) == set()
    states = pd.DataFrame(
        [s for s in decided_samples] + [s for s in speculated_samples]
    )
    actions = pd.DataFrame(
        [decided_samples[s] for s in decided_samples]
        + [speculated_samples[s] for s in speculated_samples]
    )
    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=seed,
        max_features=None,
        max_leaf_nodes=None,
    )
    model.fit(states, actions)
    assert all(
        model.predict(pd.DataFrame([state]))[0] == act
        for state, act in decided_samples.items()
    )
    assert all(
        model.predict(pd.DataFrame([state]))[0] == act
        for state, act in speculated_samples.items()
    )
    policy = lambda env: policy_decision_tree(env, model, feature_register[env_name])
    if save:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
    return policy, model


def plot_policy(
    model: DecisionTreeClassifier, class_names: Set[str], env_name: str
) -> None:
    plot_tree(
        model,
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
