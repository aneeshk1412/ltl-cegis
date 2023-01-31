#!/usr/bin/python


def ltlSpec(formula):
    return None


def completeModelCheck(action_selection_policy, specification):
    """Check if the action_selection_policy satisfies the specification
    for every possible initial state.
    Return a trace if the specification is not satisfied.
    """
    return sat, trace


def learnActionSelectionPolicy(speculated_samples, decided_samples):
    """Returns an ASP from states to actions, that generalizes
    the speculated_samples and decided_samples using an off-the-shelf
    PBD algorithm.
    Return False if there is a contradiction between the sample sets.
    """
    return action_selection_policy


def correctTrace(
    action_selection_policy,
    speculated_samples,
    decided_samples,
    specification,
    working_counter_examples_list,
    corrected_counter_examples_list,
    remaining_actions,
):
    """Try to correct a single trace out of the working_counter_examples_list
    and propagate all samples if corrected.
    Returns a dictionary of new speculated_samples.

    TODO:
    - If there is a state 's' in any of the counter examples, for which, for all envs seen till now,
    taking action 'a' leads to a state with a known safe path, assign 's' -> 'a' in speculated_samples.
    """
    tau_prime = selectTraceToCorrect(working_counter_examples_list)
    for i in Order(tau_prime):
        e_i, s_i, a_i = tau_prime[i]
        for a in next(remaining_actions[s_i]):
            speculated_samples[s_i] = a
            intermediate_asp = learnActionSelectionPolicy(speculated_samples, decided_samples)
            if doesNotSatisfy(intermediate_asp, corrected_counter_examples_list):
                continue
            ## TODO: what other rejection tests to use
    return speculated_samples, remaining_actions


def CEGIS():
    demonstrations = []

    speculated_samples = dict()  # states -> actions
    decided_samples = dict()  # states -> actions

    for tau in demonstrations:
        for e, s, a in tau:
            decided_samples[s] = a

    remaining_actions = dict()  # states -> list(actions)

    working_counter_examples_list = []
    corrected_counter_examples_list = []

    specification = ltlSpec(ltl_formula)

    while True:
        action_selection_policy = learnActionSelectionPolicy(
            speculated_samples, decided_samples
        )
        sat, trace = completeModelCheck(action_selection_policy, specification)
        if sat:
            return action_selection_policy

        working_counter_examples_list.append(trace)
        speculated_samples, remaining_actions = correctTrace(
            action_selection_policy,
            speculated_samples,
            decided_samples,
            working_counter_examples_list,
            corrected_counter_examples_list,
            remaining_actions,
        )
