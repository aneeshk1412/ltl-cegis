#!/usr/bin/env python3

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import DIR_TO_VEC

from typing import Tuple, List
from collections import deque


def sign(v):
    return v // abs(v) if v != 0 else 0


def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))


def sub(u, v):
    return tuple(ui - vi for ui, vi in zip(u, v))


def bfs(env: MiniGridEnv, obj: str):
    q = deque()
    q.append(env.agent_pos)
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
        elif c.type == obj:
            return True, x
    return False, None


##
#   DSL Terms and Functions
##


def is_present(env: MiniGridEnv, obj: str):
    b, _ = bfs(env, obj)
    return b


def get_nearest(env: MiniGridEnv, obj: str):
    _, obj_pos = bfs(env, obj)
    return obj_pos


def check(env: MiniGridEnv, pos: Tuple[int, ...], obj: str):
    c = env.grid.get(*pos)
    if c is None:
        return False
    if c.type == "door" and c.is_open:
        return False
    return c.type == obj


def is_agent_facing(env: MiniGridEnv, pos: Tuple[int, ...]):
    if pos is None:
        return False
    return dot(sub(pos, env.agent_pos), DIR_TO_VEC[env.agent_dir]) > 0

def is_at_agents_left(env: MiniGridEnv, pos: Tuple[int, ...]):
    if not pos:
        return False
    diff_vec = sub(pos, env.agent_pos)
    dir_vec = DIR_TO_VEC[env.agent_dir]
    return ( diff_vec[1]*dir_vec[0] < 0) or ( diff_vec[0]*dir_vec[1] > 0)

def is_at_agents_right(env: MiniGridEnv, pos: Tuple[int, ...]):
    if not pos:
        return False
    diff_vec = sub(pos, env.agent_pos)
    dir_vec = DIR_TO_VEC[env.agent_dir]
    return (diff_vec[1]*dir_vec[0] > 0) or (diff_vec[0]*dir_vec[1] < 0)


##
#   Features for Decision Trees
##


def extract_features_DoorKey(env: MiniGridEnv) -> Tuple[bool, ...]:
    features = (
        is_present(env, "goal"),
        is_agent_facing(env, get_nearest(env, "goal")),
        is_at_agents_left(env, get_nearest(env, "goal")),
        is_at_agents_right(env, get_nearest(env, "goal")),
        check(env, env.front_pos, "goal"),

        is_present(env, "door"),
        is_agent_facing(env, get_nearest(env, "door")),
        is_at_agents_left(env, get_nearest(env, "door")),
        is_at_agents_right(env, get_nearest(env, "door")),
        check(env, env.front_pos, "door"),

        is_present(env, "key"),
        is_agent_facing(env, get_nearest(env, "key")),
        is_at_agents_left(env, get_nearest(env, "key")),
        is_at_agents_right(env, get_nearest(env, "key")),
        check(env, env.front_pos, "key"),

        check(env, env.front_pos, "empty"),
        check(env, env.front_pos, "wall"),
    )
    return features



def env_state_to_readable_str(env:MiniGridEnv):
    res = ''
    i = 0 
    indent = '  '
    vals = extract_features_DoorKey(env)
    for key in feature_headers_DoorKey():
        key = key.replace('env, ', '').replace('"','').replace('env.','').replace('get_nearest(','').replace('check(front_pos, ','in_front(').replace('is_','').replace('agent_','').replace('))',')').replace('agent_','').replace('agents_','')
        res += indent
        res += ('F'+str(i)+":").ljust(5,' ')
        res += (key+':').ljust(17,' ')
        res += str(vals[i])
        res += '\n'
        i+=1
    return res


def readable_headers_list():
    return [key.replace('env, ', '').replace('"','').replace('env.','').replace('get_nearest(','').replace('check(front_pos, ','in_front(').replace('is_','').replace('agent_','').replace('))',')').replace('agent_','').replace('agents_','') for key in feature_headers_DoorKey()]
        


def feature_headers_DoorKey() -> List[str]:
    headers = [
        'is_present(env, "goal")',
        'is_agent_facing(env, get_nearest(env, "goal"))',
        'is_at_agents_left(env, get_nearest(env, "goal"))',
        'is_at_agents_right(env, get_nearest(env, "goal"))',
        'check(env, env.front_pos, "goal")',

        'is_present(env, "door")',
        'is_agent_facing(env, get_nearest(env, "door"))',
        'is_at_agents_left(env, get_nearest(env, "door"))',
        'is_at_agents_right(env, get_nearest(env, "door"))',
        'check(env, env.front_pos, "door")',
        
        'is_present(env, "key")',
        'is_agent_facing(env, get_nearest(env, "key"))',
        'is_at_agents_left(env, get_nearest(env, "key"))',
        'is_at_agents_right(env, get_nearest(env, "key"))',
        'check(env, env.front_pos, "key")',
        
        'check(env, env.front_pos, "empty")',
        'check(env, env.front_pos, "wall")',
    ]
    return headers


feature_register = {
    "MiniGrid-DoorKey-16x16-v0": extract_features_DoorKey,
}

header_register = {
    "MiniGrid-DoorKey-16x16-v0": feature_headers_DoorKey(),
}
