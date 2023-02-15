#!/usr/bin/env python3

import pickle
import pandas as pd
from dsl_minigrid import header_register


def intersperse(l, ele, n=3):
    return [x for y in (l[i:i+n] + [ele] * (i < len(l) - (n-1)) for i in range(0, len(l), n)) for x in y]


def load_list_from_pickle(filename: str):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def demos_to_positive_samples_csv(demos, env_name: str):
    d = dict()
    for trace in demos:
        for _, s, a, _, _ in trace:
            assert not s in d or d[s] == a
            d[s] = a
    df = pd.DataFrame(
        list(set(tuple(list(s) + [a])
             for trace in demos for _, s, a, _, _ in trace)),
        columns=header_register[env_name] + tuple(["action"]),
    )
    df.to_csv(env_name + "-demos.csv", index=False)


def csv_to_positive_samples_dict(filename, env_name: str):
    df = pd.read_csv(filename)
    l = df.to_dict("records")
    l = [
        (tuple(record[key]
         for key in header_register[env_name]), record["action"])
        for record in l
    ]
    return dict(l)
