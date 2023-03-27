from typing import List
import z3

from custom_types import Features, Action
from trace import Trace
from learner import learn_policy
from verifier import verify_policy
from simulator import simulate_policy_on_list_of_states, simulate_policy_on_state

def name(b: Features, a: Action):
    return str(b) + "_" + str(a)


def z3var(name: str, mp: dict[str, z3.Bool]):
    try:
        return mp[name]
    except KeyError:
        mp[name] = z3.Bool(name)
        return mp[name]


def sat_based_cegis(demonstrations: List[Trace]):
    solver = z3.Solver()
    mp = dict()

    for demo in demonstrations:
        solver.add(z3.And([z3var(name(b, a), mp) for b, a in demo.items()]))

