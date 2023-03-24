#!/usr/bin/env python3

from typing import List, Tuple

from custom_types import Policy, State
from trace import Trace


def simulate_policy_on_state(
    policy: Policy, state: State, task: str = "minigrid", args=None
) -> Tuple[bool, Trace]:
    pass


def simulate_policy_on_list_of_states(
    policy: Policy, state_list: List[State], task: str = "minigrid", args=None
) -> List[Tuple[bool, Trace]]:
    pass
