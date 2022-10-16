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
    pass # return true of the given bexp is valid in the given state and env
    # this function will recursively call itself. For example, if the input Bexp is 
    # of the form AND(b1,b2), then we should return the following:
    # return eval_bexp(b1, State, Environment) && eval_bexp(b2, State, Environment)
    # similar for disjunction 
    # for other boolean expressions we can simply look at the environemnt or the state and decide 
    # whether return true or false



# a demonstratoin (positive or negative) is a sequence of tuples of (State, Action)
#def check_consistency(ASP, list[(State, Action)]) -> Boolean:
#    pass



def main(arguments):

    synth = Synthesizer(action_set=Action, prop_set=Prop)
    x = synth.enumerate_positions()
    x = list(map(lambda be: be.pretty_str(), x))
    y = synth.enumerate_expressions()
    y = list(map(lambda be: be.pretty_str(), y))
    z = synth.enumerate_atomic_bexps()
    z = list(map(lambda be: be.pretty_str(), z))

    t = synth.enumerate_bexps()
    t = list(map(lambda be: be.pretty_str(), t))

    i = 0
    for tt in t:
        print (str(i) + ":  "+ tt)
        i += 1




    return 
    policy_depth = 2
    asps = synth.enumerate_all_asps(depth=policy_depth, cap=500)
    number_of_asps = len(asps)
    for asp in asps:
        print(asp.pretty_str())
        print (50*'-')
    print (number_of_asps, ' ASPs of length =', policy_depth,'are generated')

        
    


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))