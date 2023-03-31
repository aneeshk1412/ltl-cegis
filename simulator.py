#!/usr/bin/env python3

def simulate_policy_on_state(
    state: State,
    policy: Policy,
    feature_func: Callable[[State], Features],
    specification: Specification,
    args=None,
) -> Tuple[bool, Trace]:
    trace = Trace()
    seen_state_id_set = set()
    s = deepcopy(state)
    while True:
        seen_state_id_set.add(s.id())

