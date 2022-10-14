#!/usr/bin/python3
__author__ = "Kia Rahmani"

from src.help import product
from src.constants import _HALLWAY_LENGTH
from src.asp import ASP, Action, enumerate_bexpressions, enumerate_expressions, enumerate_positions
import itertools


class Synthesizer:
    def __init__(self) -> None:
        print('initializing the program synthesizer')

    # returns ASPS with a single (condition, action) pair
    def enumerate_all_asps(self, depth=1, cap=100) -> list[ASP]:
        """ returns all possible ASPs within predefined bounds """
        res = []
        poss = enumerate_positions()
        #debug_poss = list(map (lambda e:e.pretty_str(), poss))
        exps = enumerate_expressions(
            limit=_HALLWAY_LENGTH, seed_positions=poss)
        #debug_exps = list(map (lambda e:e.pretty_str(), exps))
        bexps = enumerate_bexpressions(
            max_offset=5, seed_expressions=exps, seed_positions=poss)
        #debug_bexps = list(map(lambda be: be.pretty_str(), bexps))
        id = 0
        action_tuples = list(itertools.permutations(list(Action), r=depth+1))
        for action_tuple in action_tuples:
            bexp_tuples = list(product(lst=bexps, dim=depth, cap=cap))
            for bexp_tuple in bexp_tuples:
                asp = ASP(id = id, else_action=action_tuple[-1])
                id += 1
                for bexp, action in zip(bexp_tuple, action_tuple):
                    asp.add_transition_cond_pair(bexp, action)
                    res.append(asp)
        return res



    def get_next_asp():
        pass

    