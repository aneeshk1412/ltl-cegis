#!/usr/bin/python3
__author__ = "Kia Rahmani"

from src.asp import ASP, Action, enumerate_bexpressions, enumerate_expressions, enumerate_positions


class Synthesizer:
    def __init__(self) -> None:
        print('initializing the program synthesizer')
        self.next_asp = ASP()

    # returns ASPS with a single (condition, action) pair
    def enumerate_all_asps(self) -> list[ASP]:
        """ returns all possible ASPs within predefined bounds """
        res = []
        poss = enumerate_positions()
        exps = enumerate_expressions(limit=5, seed_positions=poss)
        bexps = enumerate_bexpressions(max_offset=5, seed_expressions=exps, seed_positions=poss)
        for action in Action:
          for bexp in bexps:
            asp = ASP()
            asp.add_transition_cond_pair(bexp, action)
            res.append(asp)
        return res

