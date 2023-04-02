#!/usr/bin/env python3

from typing import Tuple
from collections import deque

from commons_minigrid import State, Features

from minigrid.core.constants import DIR_TO_VEC

""" Low Level Expressions """


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


def bfs(state: State, obj: str, color: str | None = None):
    q = deque()
    if isinstance(state.agent_pos, tuple):
        q.append(state.agent_pos)
    else:
        q.append(tuple([state.agent_pos[0], state.agent_pos[1]]))
    vis = set()
    while len(q):
        x = q.popleft()
        if x in vis:
            continue
        vis.add(x)
        c = state.grid.get(x[0], x[1])
        if c is None or c.type == "door" and c.is_open:
            for d in {(1, 0), (0, 1), (0, -1), (-1, 0)}:
                q.append((x[0] + d[0], x[1] + d[1]))
        elif c.type == obj and (color is None or c.color == color):
            return True, x
    return False, None


def right_pos(state: State):
    agent_pos = state.agent_pos
    dx, dy = state.dir_vec
    return (agent_pos[0] - dy, agent_pos[1] + dx)


def left_pos(state: State):
    agent_pos = state.agent_pos
    dx, dy = state.dir_vec
    return (agent_pos[0] + dy, agent_pos[1] - dx)


""" Features """


def is_present(state: State, obj: str, color: str | None = None) -> bool:
    b, _ = bfs(state, obj, color)
    return b


def get_nearest(state: State, obj: str, color: str | None = None) -> bool:
    _, obj_pos = bfs(state, obj, color)
    return obj_pos


def check(
    state: State, pos: Tuple[int, ...], obj: str, color: str | None = None
) -> bool:
    c = state.grid.get(*pos)
    if c is None:
        return False
    if c.type == "door" and c.is_open:
        return False
    return c.type == obj and (color is None or c.color == color)


def is_agent_on(state: State, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return all(p == e for p, e in zip(pos, state.agent_pos))


def is_at_agents_front(state: State, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return dot(DIR_TO_VEC[state.agent_dir], sub(pos, state.agent_pos)) > 0


def is_at_agents_back(state: State, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return dot(DIR_TO_VEC[state.agent_dir], sub(pos, state.agent_pos)) < 0


def is_at_agents_left(state: State, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return cross(DIR_TO_VEC[state.agent_dir], sub(pos, state.agent_pos)) < 0


def is_at_agents_right(state: State, pos: Tuple[int, ...]) -> bool:
    if not pos:
        return False
    return cross(DIR_TO_VEC[state.agent_dir], sub(pos, state.agent_pos)) > 0


""" Feature Functions """


def main_object_features(state: State, *args) -> Features:
    obj_name = "_".join(args)
    return {
        f"is_present__{obj_name}": is_present(state, *args),
        f"is_agent_on__{obj_name}": is_agent_on(state, get_nearest(state, *args)),
        f"is_at_agents_front__{obj_name}": is_at_agents_front(state, get_nearest(state, *args)),
        f"is_at_agents_back__{obj_name}": is_at_agents_back(state, get_nearest(state, *args)),
        f"is_at_agents_left__{obj_name}": is_at_agents_left(state, get_nearest(state, *args)),
        f"is_at_agents_right__{obj_name}": is_at_agents_right(state, get_nearest(state, *args)),
        f"check_agent_front_pos__{obj_name}": check(state, state.front_pos, *args),
        f"check_agent_left_pos__{obj_name}": check(state, left_pos(state), *args),
        f"check_agent_right_pos__{obj_name}": check(state, right_pos(state), *args),
    }


def other_object_features(state: State, *args) -> Features:
    obj_name = "_".join(args)
    return {
        f"check_agent_front_pos__{obj_name}": check(state, state.front_pos, *args),
        f"check_agent_left_pos__{obj_name}": check(state, left_pos(state), *args),
        f"check_agent_right_pos__{obj_name}": check(state, right_pos(state), *args),
    }


def features_empty(state: State) -> Features:
    return {
        **main_object_features(state, "goal"),
        **other_object_features(state, "wall"),
        f"check_agent_front_pos__empty": check(state, state.front_pos, "empty")
    }


""" Env Name to Feature Function Mapping """

feature_mapping = {
    "MiniGrid-Empty-Random-6x6-v0": features_empty,
}
