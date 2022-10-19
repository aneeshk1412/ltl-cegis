#!/usr/bin/python3
__author__ = "Kia Rahmani"

from src.dsl import Prop, ASP, Action
from src.dsl import Position, Expression, BoolExp, AtomicBoolExp
import itertools


class Synthesizer:
    def __init__(self, action_set, prop_set) -> None:
        # how many conjuncts to be considered for bexps, i.e. 1 means A /\ B is considered by A /\ B /\ C is not
        self.max_conjunction_depth = 1
        self.max_disjunction_depth = 1  # similar to conjunction depth defined above
        self.actions = action_set
        self.props = prop_set
        self.agent_cnt = 0
        self.hallway_length = 5
        self.enumerated_positions = None
        self.enumerated_expressions = None
        self.enumerated_atomic_bexps = None
        self.enumerated_bexps = None

    def enumerate_positions(self):
        res = [Position(tp='robot_pos', children=[])]
        for a in range(self.agent_cnt):
            res.append(Position(tp='agent_pos', children=[a]))
        self.enumerated_positions = res
        return res

    def enumerate_expressions(self):
        if not self.enumerated_positions:
            self.enumerate_positions()
        res = []
        for pos in self.enumerated_positions:
            res.append(Expression(tp='from_pos', children=[pos]))
        for c in range(self.hallway_length):
            res.append(Expression(tp='from_int', children=[c]))

        for p1 in self.enumerated_positions:
            for p2 in self.enumerated_positions:
                if p1 == p2:
                    continue
                res.append(Expression(tp='diff', children=[p1, p2]))
        self.enumerated_expressions = res
        return res

    def enumerate_atomic_bexps(self):
        if not self.enumerated_positions:
            self.enumerate_positions()
        if not self.enumerated_expressions:
            self.enumerate_expressions()
        res = []
        for action in Action:
            res.append(AtomicBoolExp(tp='curr_rob_act', children=[action]))
        for b in {True, False}:
            res.append(AtomicBoolExp(tp='from_bool', children=[b]))
        for op in {'lt', 'gt', 'eq'}:
            for (e1, e2) in list(itertools.combinations(self.enumerated_expressions, 2)):
                if e1.tp == 'from_int' and e2.tp == 'from_int':
                    continue
                res.append(AtomicBoolExp(tp=op, children=[e1, e2]))
        for prop in Prop:
            for pos in self.enumerated_positions:
                for offset in range(self.hallway_length):
                    res.append(AtomicBoolExp(tp='check_prop',
                               children=[pos, prop, offset]))
        self.enumerated_atomic_bexps = res
        return res

    def enumerate_bexps(self):
        if not self.enumerated_positions:
            self.enumerate_positions()
        if not self.enumerated_expressions:
            self.enumerate_expressions()
        if not self.enumerated_atomic_bexps:
            self.enumerate_atomic_bexps()
        res = self.enumerated_atomic_bexps
        for (b1, b2) in list(itertools.combinations(self.enumerated_atomic_bexps, 2)):
            if b1.tp == 'from_bool' or b2.tp == 'from_bool':
                continue
            for op in {'or', 'and'}:

                res.append(BoolExp(tp=op, children=[b1, b2]))
        self.enumerated_bexps = res
        return res

    def enumerate_asps(self, cap=10000):
        if not self.enumerated_bexps:
            self.enumerate_bexps()
        result = []
        action_perms = list(itertools.permutations(self.actions, 3))
        condition_combs = list(
            itertools.combinations(self.enumerated_bexps, 2))
        id = 0
        flag = True
        for a1, a2, a3 in action_perms:
            if not flag:
                break
            for b1, b2 in condition_combs:
                result.append(ASP(id=id, transition_cond_pairs=[
                    (b1, a1), (b2, a2)], else_action=a3))
                id += 1
                # if id > cap:
                #    flag = False
                #    break
        return result
