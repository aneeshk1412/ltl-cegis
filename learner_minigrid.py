#!/usr/bin/env python3

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import Set, Tuple, List
from sklearn.tree import DecisionTreeClassifier, plot_tree

from commons_minigrid import Decisions, debug, Features, Policy, Action


def augment_policy(policy: Policy, decisions: Decisions) -> Policy:
    def new_policy(feats: Features) -> Action:
        if feats in decisions.features_to_actions:
            return list(decisions.features_to_actions[feats])[0]
        else:
            return policy(feats)

    return new_policy


def policy_decision_tree(model: DecisionTreeClassifier) -> Policy:
    def policy(feats: Features) -> Action:
        df = pd.DataFrame([feats.id])
        return model.predict(df)[0]

    return policy


def learn(
    decisions: Decisions,
    seed: int | None = None,
    save: bool = False,
) -> Tuple[Policy, DecisionTreeClassifier]:
    keys_list, actions_list = zip(*decisions.get_decisions())
    keys_df = pd.DataFrame(keys_list)
    actions_df = pd.DataFrame(actions_list)
    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=seed,
        max_features=None,
        max_leaf_nodes=None,
    )
    model.fit(keys_df, actions_df)
    agrees = sum(
        model.predict(pd.DataFrame([key]))[0] == action
        for key, action in zip(keys_list, actions_list)
    )
    debug(f"Percent Agreement: {agrees / len(actions_list)}")
    policy = policy_decision_tree(model)
    if save:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
    return policy, model


def plot_model(
    model: DecisionTreeClassifier, class_names: Set[str], feature_names: List[str]
) -> None:
    plot_tree(
        model,
        max_depth=None,
        class_names=class_names,
        label="none",
        precision=1,
        feature_names=feature_names,
        rounded=True,
        fontsize=5,
        proportion=True,
    )
    plt.show()
