#!/usr/bin/python3
__coauthor__ = ["Kia Rahmani", "Aneesh Shetty"]


"""A simple test program which repeatedly creates and prints programs using a synthesizer """
import sys
from typing import Set
from synthesizer.synthesize import *
from synthesizer.dsl import *
from verifier.verify import verifies

# a state contains the values of all state variables
# it is equipped with a series of functions to evaluate given expressions in this state


class State:
    def __init__(self, ra, rpos, aposs, prop_map):
        self.robot_action = ra  # the robot's current action
        self.robot_position = rpos  # an int representing the position of the robot
        # a list where index=i contains the position (an int value) of agent_i
        self.agent_positions = aposs
        # a dictionary from Prop to a set of indexes where the prop is valid
        self.prop_map = prop_map

    # a function which returns true if prop holds at index i in this state
    def check_prop_at_position(self, prop: Prop, index: int) -> bool:
        return index in self.prop_map.get(prop)

    # function which returns the current values of robot/agent position in this state
    def eval_pos(self, pos: Position) -> int:
        if pos.tp == 'robot_pos':
            return self.robot_position
        elif pos.tp == 'agent_pos':
            i = pos.children[0]  # agent_i's position should be returned
            return self.agent_positions[i]
        else:
            raise Exception('unknown position type')

    # function which evaluates a given expression and returns the corresponding int value in this state
    def eval_exp(self, exp: Expression) -> int:
        if exp.tp == 'from_pos':
            pos = exp.children[0]
            return self.eval_pos(pos)
        elif exp.tp == 'from_int':
            return exp.children[0]
        elif exp.tp == 'diff':
            pos1 = exp.children[0]
            pos2 = exp.children[1]
            return self.eval_pos(pos1) - self.eval_pos(pos2)
        else:
            raise Exception('unknown expression type')

    # return true if the given atomic boolean expression is valid in this state
    def eval_atomic_bexp(self, abexp: AtomicBoolExp) -> bool:
        if abexp.tp == 'from_bool':
            return abexp.children[0]
        elif abexp.tp == 'curr_rob_act':
            return self.robot_action == abexp.children[0]
        elif abexp.tp == 'check_prop':
            pos = abexp.children[0]
            prop = abexp.children[1]
            offset = abexp.children[2]
            return self.check_prop_at_position(prop, self.eval_pos(pos)+offset)
        elif abexp.tp == 'lt':
            exp1 = abexp.children[0]
            exp2 = abexp.children[1]
            return self.eval_exp(exp1) < self.eval_exp(exp2)
        elif abexp.tp == 'eq':
            exp1 = abexp.children[0]
            exp2 = abexp.children[1]
            return self.eval_exp(exp1) == self.eval_exp(exp2)
        elif abexp.tp == 'gt':
            exp1 = abexp.children[0]
            exp2 = abexp.children[1]
            return self.eval_exp(exp1) > self.eval_exp(exp2)
        else:
            raise Exception('unknown atomic boolean expression type')

    # return true of the given bexp is valid in this state
    def eval_bexp(self, bexp: BoolExp) -> bool:
        if bexp.tp == 'and':
            b0 = bexp.children[0]
            b1 = bexp.children[1]
            return self.eval_atomic_bexp(b0) and self.eval_atomic_bexp(b1)
        elif bexp.tp == 'or':
            b0 = bexp.children[0]
            b1 = bexp.children[1]
            return self.eval_atomic_bexp(b0) or self.eval_atomic_bexp(b1)
        else:
            raise Exception('unknown boolean expression type')


class Demonstration:
    def __init__(self, demo_tp, state_action_transitions) -> None:
        assert demo_tp in {'pos', 'neg'}
        self.demo_tp = demo_tp  # positive demonstration or negative demonstration?
        # a list of tuples [(s_1,a_1),(s_2,a_2),...,(s_k,a_k)]
        self.state_action_transitions = state_action_transitions

    # for a given action A, return the set of all states s such that (s, A) is a tuple in this demonstration
    def get_states_for_action(self, action: Action):
        res = set()
        for s, a in self.state_action_transitions:
            if a == action:
                res.add(s)
        return res


# this function takes a demonstration and an ASP and returns 
# true if the ASP is consistent with the demo (i.e. running the 
# ASP starting from the first state would result in the same sequence of actions being taken)
def check_demo_consistency(self, asp: ASP, demo: Demonstration):
    for s, a in demo.state_action_transitions:
        asp_predicate_for_act = asp.get_predicate_for_action(a)
        if not s.eval_bexp(asp_predicate_for_act):
            return False
    return True


def main(arguments):
    synth = Synthesizer(action_set=Action, prop_set=Prop)
    asp_list = synth.enumerate_asps(cap=1000)
    print('>>>', str(len(asp_list)), ' ASPs of length =',
          len(synth.actions), 'are generated')

    i = 0
    for iter in range(len(asp_list)):
        input('>>> check the next 100 ASPs?\n\n')
        for j in range(100):
            print("Verifying: ")
            print(cstr(asp_list[i]))
            print()
            print(verifies(cstr(asp_list[i])))
            i += 1
            print(50*'-')
    return


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

# the ultimate goal is to synthesize a predicate p_i for
# action a_i that the asp can execute
# a demonstration D provides a set of state  for
# each action a_i(under which a_i was taken in D)

# if the demonstration is positive: we have to make sure P_i is
# consistent with `all` samples
