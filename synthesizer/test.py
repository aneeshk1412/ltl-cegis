#!/usr/bin/python3
__author__ = "Kia Rahmani"


"""A simple test program which repeatedly creates and prints programs using a synthesizer """
import sys
from src.synthesizer import *


class Environment:
    def __init__(self):
        # keep an internal data structure (say, a map from positions to the set of props
        # that are valid in that posision)
        pass

    def check_prop_at_position(Pos, Prop):
        # simpoly look at the data structure and return the result
        pass


class State:
    def __init__(self, ra):
        self.robot_action = ra
        # ... similar for agents and robot position


def eval_bexp(Bexp, State, Environment):
    pass  # return true of the given bexp is valid in the given state and env
    # this function will recursively call itself. For example, if the input Bexp is
    # of the form AND(b1,b2), then we should return the following:
    # return eval_bexp(b1, State, Environment) && eval_bexp(b2, State, Environment)
    # similar for disjunction
    # for other boolean expressions we can simply look at the environemnt or the state and decide
    # whether return true or false


# a demonstratoin (positive or negative) is a sequence of tuples of (State, Action)
# def check_consistency(ASP, list[(State, Action)]) -> Boolean:
#    pass


def main(arguments):

    synth = Synthesizer(action_set=Action, prop_set=Prop)
    asp_list = synth.enumerate_asps(cap=1000) 
    print('>>>',str(len(asp_list)), ' ASPs of length =', len(synth.actions), 'are generated')
    
    i = 0 
    for iter in range(len(asp_list)):
        input('>>> print the next 100 ASPs?\n\n')
        for j in range(100):
            print (asp_list[i].pretty_str())
            i += 1
            print(50*'-')
    return 


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
