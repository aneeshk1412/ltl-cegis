#!/usr/bin/python3
__author__ = "Kia Rahmani"

from re import A
import string
from unittest import result
from src.help import product
from src.constants import _HALLWAY_LENGTH
from src.asp import Prop, ASP, Action
from src.help import bcolors
import itertools


class Position:
    def __init__(self, tp: string, children):
        assert (tp in {'robot_pos', 'agent_pos'})
        if (tp == 'robot_pos'):
            (len(children) == 0)
        else:
            assert (len(children) == 1)
        self.tp = tp
        self.children = children

    def pretty_str(self) -> string:
        if self.tp == 'robot_pos':
            return 'R#Pos'
        elif self.tp == 'agent_pos':
            return 'A'+str(self.children[0])+'#Pos'
        else:
            raise Exception("unexpected position type")


class Expression:
    def __init__(self, tp: string, children) -> None:
        assert (tp in {'from_pos', 'from_int', 'diff'})
        if tp == 'diff':
            assert (len(children) == 2)
        else:
            assert (len(children) == 1)
        self.tp = tp
        self.children = children

    def pretty_str(self) -> string:
        if self.tp == 'from_pos':
            return self.children[0].pretty_str()
        elif self.tp == 'from_int':
            return ''+str(self.children[0])+''
        elif self.tp == 'diff':
            return 'diff(' + self.children[0].pretty_str() + ',' + \
                self.children[1].pretty_str() + ')'
        else:
            raise Exception("unexpected expression type")


class BoolExp:
    def __init__(self, tp: string, children) -> None:
        assert (tp in {'and', 'or'})
        assert (len(children) == 2)
        self.tp = tp
        self.children = children

    def pretty_str(self) -> string:
        if self.tp == 'and':
            op = '∧'
        elif self.tp == 'or':
            op = '∨'
        return '(' + self.children[0].pretty_str() + ' ' + op + ' ' + \
            self.children[1].pretty_str() + ')'


class AtomicBoolExp:
    def __init__(self, tp: string, children) -> None:
        assert (tp in {'from_bool', 'check_prop',
                'curr_rob_act', 'lt', 'gt', 'eq'})
        if tp == 'from_bool':
            assert (len(children) == 1)
        if tp == 'check_prop':
            assert (len(children) == 3)  # pos,prop,offset
        if tp == 'curr_rob_act':
            assert (len(children) == 1)
        if tp in {'lt', 'gt', 'eq'}:
            assert (len(children) == 2)
        self.tp = tp
        self.children = children

    def pretty_str(self) -> string:
        res = ''
        if self.tp == 'from_bool':
            res = str(self.children[0])
        elif self.tp == 'curr_rob_act':
            res = 'cuurent_action_is('+str(self.children[0])+')'
        elif self.tp in {'lt', 'gt', 'eq'}:
            res = self.tp + \
                "("+self.children[0].pretty_str() + \
                ',' + self.children[1].pretty_str()+')'
        elif self.tp == 'check_prop':
            res = 'check_' + \
                str(self.children[1]) + '(' + self.children[0].pretty_str() + \
                ',' + str(self.children[2]) + ')'
        else:
            raise Exception("unexpected boolean expression type")
        return bcolors.FAIL + res + bcolors.ENDC
        # return res


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
                #if id > cap:
                #    flag = False
                #    break
        return result
