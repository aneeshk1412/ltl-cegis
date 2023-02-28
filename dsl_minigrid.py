#!/usr/bin/env python3

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import DIR_TO_VEC

from typing import Tuple
from collections import deque


def sign(v) -> int:
    return v // abs(v) if v != 0 else 0


def dot(u, v):
    # basically |u||v| cos(theta)
    return sum(ui * vi for ui, vi in zip(u, v))


def cross(u, v):
    # basically |u||v| sin(theta)
    return u[0] * v[1] - u[1] * v[0]


def sub(u, v):
    return tuple(ui - vi for ui, vi in zip(u, v))


def bfs(env: MiniGridEnv, obj: str, color: str | None = None):
    q = deque()
    if isinstance(env.agent_pos, tuple):
        q.append(env.agent_pos)
    else:
        q.append(tuple([env.agent_pos[0], env.agent_pos[1]]))
    vis = set()
    while len(q):
        x = q.popleft()
        if x in vis:
            continue
        vis.add(x)
        c = env.grid.get(x[0], x[1])
        if c is None or c.type == "door" and c.is_open:
            for d in {(1, 0), (0, 1), (0, -1), (-1, 0)}:
                q.append((x[0] + d[0], x[1] + d[1]))
        elif c.type == obj and (color is None or c.color == color):
            return True, x
    return False, None


##
#   DSL Terms and Functions
##


def is_present(env: MiniGridEnv, obj: str, color: str | None = None) -> bool:
    b, _ = bfs(env, obj, color)
    return b


def get_nearest(env: MiniGridEnv, obj: str, color: str | None = None) -> bool:
    _, obj_pos = bfs(env, obj, color)
    return obj_pos


def check(env: MiniGridEnv, pos: Tuple[int, ...], obj: str, color: str | None = None) -> bool:
    c = env.grid.get(*pos)
    if c is None:
        return False
    if c.type == "door" and c.is_open:
        return False
    return c.type == obj and (color is None or c.color == color)


def is_agent_on(env: MiniGridEnv, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return all(p == e for p, e in zip(pos, env.agent_pos))


def is_agent_facing(env: MiniGridEnv, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return dot(DIR_TO_VEC[env.agent_dir], sub(pos, env.agent_pos)) > 0


def is_at_agents_left(env: MiniGridEnv, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return cross(DIR_TO_VEC[env.agent_dir], sub(pos, env.agent_pos)) < 0


def is_at_agents_right(env: MiniGridEnv, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return cross(DIR_TO_VEC[env.agent_dir], sub(pos, env.agent_pos)) > 0


##
#   Features for Decision Trees
##

def common_features(env: MiniGridEnv, *args):
    return (
        is_present(env, *args),
        is_agent_facing(env, get_nearest(env, *args)),
        is_at_agents_left(env, get_nearest(env, *args)),
        is_at_agents_right(env, get_nearest(env, *args)),
        check(env, env.front_pos, *args),
    )

def common_headers(*args):
    val = '"' + '", "'.join(args) + '"'
    return (
        f'is_present({val})',
        f'is_agent_facing(get_nearest({val}))',
        f'is_at_agents_left(get_nearest({val}))',
        f'is_at_agents_right(get_nearest({val}))',
        f'check(front_pos, {val})',
    )

def features_Empty(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        *common_features(env, "goal"),
    )
    return features

def header_Empty() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        *common_headers("goal"),
    )

def features_LavaGap(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        check(env, env.front_pos, "lava"),
        *common_features(env, "goal"),
    )
    return features

def header_LavaGap() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        'check(front_pos, "lava")',
        *common_headers("goal"),
    )

def features_DoorKey(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        *common_features(env, "goal"),
        *common_features(env, "door"),
        *common_features(env, "key"),
    )
    return features

def header_DoorKey() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        *common_headers("goal"),
        *common_headers("door"),
        *common_headers("key"),
    )

def features_MultiKeyDoorKey_1(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        *common_features(env, "goal"),
        *common_features(env, "door"),
        *common_features(env, "key", "red"),
    )
    return features

def header_MultiKeyDoorKey_1() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        *common_headers("goal"),
        *common_headers("door"),
        *common_headers("key", "red"),
    )

def progress_MultiKeyDoorKey_1(env: MiniGridEnv) -> int:
    num_keys_remaining = sum(is_present(env, "key", c) for c in {'green'})
    if num_keys_remaining == 1:
        return 0
    ## All keys picked
    if not is_present(env, "goal"):
        return 1
    ## Door is opened
    if not is_agent_on(env, get_nearest(env, "goal")):
        return 2
    return 3


def features_MultiKeyDoorKey_2(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        *common_features(env, "goal"),
        *common_features(env, "door"),
        *common_features(env, "key", "red"),
        *common_features(env, "key", "green"),
    )
    return features

def header_MultiKeyDoorKey_2() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        *common_headers("goal"),
        *common_headers("door"),
        *common_headers("key", "red"),
        *common_headers("key", "green"),
    )

def progress_MultiKeyDoorKey_2(env: MiniGridEnv) -> int:
    num_keys_remaining = sum(is_present(env, "key", c) for c in {'green'})
    if num_keys_remaining == 2:
        return 0
    if num_keys_remaining == 1:
        return 1
    ## Both keys picked
    if not is_present(env, "goal"):
        return 2
    ## Door is opened
    if not is_agent_on(env, get_nearest(env, "goal")):
        return 3
    return 4

def features_MultiKeyDoorKey_3(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        *common_features(env, "goal"),
        *common_features(env, "door"),
        *common_features(env, "key", "red"),
        *common_features(env, "key", "green"),
        *common_features(env, "key", "blue"),
    )
    return features

def header_MultiKeyDoorKey_3() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        *common_headers("goal"),
        *common_headers("door"),
        *common_headers("key", "red"),
        *common_headers("key", "green"),
        *common_headers("key", "blue"),
    )

def progress_MultiKeyDoorKey_3(env: MiniGridEnv) -> int:
    num_keys_remaining = sum(is_present(env, "key", c) for c in {'green'})
    if num_keys_remaining == 3:
        return 0
    if num_keys_remaining == 2:
        return 1
    if num_keys_remaining == 1:
        return 2
    ## Both keys picked
    if not is_present(env, "goal"):
        return 3
    ## Door is opened
    if not is_agent_on(env, get_nearest(env, "goal")):
        return 4
    return 5

def features_MultiKeyDoorKey_4(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        *common_features(env, "goal"),
        *common_features(env, "door"),
        *common_features(env, "key", "red"),
        *common_features(env, "key", "green"),
        *common_features(env, "key", "blue"),
        *common_features(env, "key", "purple"),
    )
    return features

def header_MultiKeyDoorKey_4() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        *common_headers("goal"),
        *common_headers("door"),
        *common_headers("key", "red"),
        *common_headers("key", "green"),
        *common_headers("key", "blue"),
        *common_headers("key", "purple"),
    )

def progress_MultiKeyDoorKey_4(env: MiniGridEnv) -> int:
    num_keys_remaining = sum(is_present(env, "key", c) for c in {'green'})
    if num_keys_remaining == 4:
        return 0
    if num_keys_remaining == 3:
        return 1
    if num_keys_remaining == 2:
        return 2
    if num_keys_remaining == 1:
        return 3
    ## All keys picked
    if not is_present(env, "goal"):
        return 4
    ## Door is opened
    if not is_agent_on(env, get_nearest(env, "goal")):
        return 5
    return 6

def features_MultiKeyDoorKey_5(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        *common_features(env, "goal"),
        *common_features(env, "door"),
        *common_features(env, "key", "red"),
        *common_features(env, "key", "green"),
        *common_features(env, "key", "blue"),
        *common_features(env, "key", "purple"),
        *common_features(env, "key", "yellow"),
    )
    return features

def header_MultiKeyDoorKey_5() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        *common_headers("goal"),
        *common_headers("door"),
        *common_headers("key", "red"),
        *common_headers("key", "green"),
        *common_headers("key", "blue"),
        *common_headers("key", "purple"),
        *common_headers("key", "yellow"),
    )

def progress_MultiKeyDoorKey_5(env: MiniGridEnv) -> int:
    num_keys_remaining = sum(is_present(env, "key", c) for c in {'green'})
    if num_keys_remaining == 5:
        return 0
    if num_keys_remaining == 4:
        return 1
    if num_keys_remaining == 3:
        return 2
    if num_keys_remaining == 2:
        return 3
    if num_keys_remaining == 1:
        return 4
    ## All keys picked
    if not is_present(env, "goal"):
        return 5
    ## Door is opened
    if not is_agent_on(env, get_nearest(env, "goal")):
        return 6
    return 7

def features_BlockedUnlockPickup(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_agent_on(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
        *common_features(env, "goal"),
        *common_features(env, "door"),
        *common_features(env, "key"),
        *common_features(env, "ball"),
    )
    return features

def header_BlockedUnlockPickup() -> Tuple[str, ...]:
    return (
        'is_agent_on(get_nearest("goal"))',
        'check(front_pos, "empty")',
        'check(front_pos, "wall")',
        *common_headers("goal"),
        *common_headers("door"),
        *common_headers("key"),
        *common_headers("ball"),
    )

header_register = {
    "MiniGrid-Empty-Random-6x6-v0": header_Empty(),
    "MiniGrid-LavaGapS7-v0": header_LavaGap(),
    "MiniGrid-DoorKey-16x16-v0": header_DoorKey(),
    "MiniGrid-MultiKeyDoorKey-16x16-1": header_MultiKeyDoorKey_1(),
    "MiniGrid-MultiKeyDoorKey-16x16-2": header_MultiKeyDoorKey_2(),
    "MiniGrid-MultiKeyDoorKey-16x16-3": header_MultiKeyDoorKey_3(),
    "MiniGrid-MultiKeyDoorKey-16x16-4": header_MultiKeyDoorKey_4(),
    "MiniGrid-MultiKeyDoorKey-16x16-5": header_MultiKeyDoorKey_5(),
    "MiniGrid-BlockedUnlockPickup-v0": header_BlockedUnlockPickup(),
}

feature_register = {
    "MiniGrid-Empty-Random-6x6-v0": features_Empty,
    "MiniGrid-LavaGapS7-v0": features_LavaGap,
    "MiniGrid-DoorKey-16x16-v0": features_DoorKey,
    "MiniGrid-MultiKeyDoorKey-16x16-1": features_MultiKeyDoorKey_1,
    "MiniGrid-MultiKeyDoorKey-16x16-2": features_MultiKeyDoorKey_2,
    "MiniGrid-MultiKeyDoorKey-16x16-3": features_MultiKeyDoorKey_3,
    "MiniGrid-MultiKeyDoorKey-16x16-4": features_MultiKeyDoorKey_4,
    "MiniGrid-MultiKeyDoorKey-16x16-5": features_MultiKeyDoorKey_5,
    "MiniGrid-BlockedUnlockPickup-v0": features_BlockedUnlockPickup,
}

progress_register = {
    # "MiniGrid-Empty-Random-6x6-v0": progress_Empty,
    # "MiniGrid-LavaGapS7-v0": progress_LavaGap,
    # "MiniGrid-DoorKey-16x16-v0": progress_DoorKey,
    "MiniGrid-MultiKeyDoorKey-16x16-1": progress_MultiKeyDoorKey_1,
    "MiniGrid-MultiKeyDoorKey-16x16-2": progress_MultiKeyDoorKey_2,
    "MiniGrid-MultiKeyDoorKey-16x16-3": progress_MultiKeyDoorKey_3,
    "MiniGrid-MultiKeyDoorKey-16x16-4": progress_MultiKeyDoorKey_4,
    "MiniGrid-MultiKeyDoorKey-16x16-5": progress_MultiKeyDoorKey_5,
    # "MiniGrid-BlockedUnlockPickup-v0": progress_BlockedUnlockPickup,
}
