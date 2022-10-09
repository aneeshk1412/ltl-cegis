#!/usr/bin/python3
__author__ = "Kia Rahmani"


from random import seed
from tokenize import String
from enum import Enum
from typing import List
from src.constants import _NUMBER_OF_AGENTS, _HALLWAY_LENGTH


class Prop(Enum):
    WALL = 0
    CHECKPOINT = 1
    TILE = 2


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    NONE = 2


class ASP:
    def __init__(self) -> None:
        self.transition_cond_pairs = set()

    def add_transition_cond_pair(self, b, a):
        self.transition_cond_pairs.add((b, a))

    def pretty_str(self) -> String:
        res = ''
        if_stmt = 'if'
        for (b, a) in self.transition_cond_pairs:
            res += if_stmt+'(' + b.pretty_str() + \
                ') then take ' + str(a) + ';\n'
            if_stmt = 'elif'
        return res


class Pos:
    def __init__(self, tp, children) -> None:
        assert (tp in {'from_int', 'robot_pos', 'agent_pos'})
        if (tp == 'robot_pos'):
            (len(children) == 0)
        else:
            assert (len(children) == 1)
        self.tp = tp
        self.children = children

    def pretty_str(self) -> String:
        if self.tp == 'from_int':
            return 'Pos(' + str(self.children[0]) + ')'
        elif self.tp == 'robot_pos':
            return 'R#Pos'
        elif self.tp == 'agent_pos':
            return 'A'+str(self.children[0])+'#Pos'
        else:
            raise Exception("unexpected position type")


class Exp:
    def __init__(self, tp, children) -> None:
        assert (tp in {'from_pos', 'from_int', 'absolute', 'bin_op'})
        if tp == 'bin_op':
            assert (len(children) == 3)
            assert (children[0] in {'+', '-'})
        else:
            assert (len(children) == 1)
        self.tp = tp
        self.children = children

    def pretty_str(self) -> String:
        if self.tp == 'from_pos':
            return self.children[0].pretty_str()
        elif self.tp == 'from_int':
            return 'Exp('+str(self.children[0])+')'
        elif self.tp == 'absolute':
            return 'abs(' + self.children[0].pretty_str() + ')'
        elif self.tp == 'bin_op':
            return self.children[1].pretty_str() + \
                str(self.children[0]) + self.children[2].pretty_str()
        else:
            raise Exception("unexpected expression type")


class BExp:
    def __init__(self, tp, children) -> None:
        assert (tp in {'check_prop', 'bin_op'})
        assert (len(children) == 3)
        if tp == 'bin_op':
            assert (children[0] in {'and', 'or', 'eq', 'lt', 'gt'})
        self.tp = tp
        self.children = children

    def pretty_str(self) -> String:
        if self.tp == 'bin_op':
            return self.children[0] + '(' + self.children[1].pretty_str() + ',' + self.children[2].pretty_str() + ')'
        elif self.tp == 'check_prop':
            return 'check_' + str(self.children[0]).replace('Prop.', '').lower() + \
                '(at=' + self.children[1].pretty_str() + \
                ', offset=' + str(self.children[2]) + ')'
        else:
            raise Exception("unexpected boolean expression type")


def enumerate_positions() -> list[Pos]:
    res = [Pos('robot_pos', [])]
    for a in range(_NUMBER_OF_AGENTS):
        res.append(Pos('agent_pos', [a]))
    for i in range(_HALLWAY_LENGTH):
        res.append(Pos('from_int', [i]))
    return res


def enumerate_expressions(limit=5, seed_positions=[]) -> list[Exp]:
    res = []
    for p in seed_positions:
        res.append(Exp('from_pos', [p]))
    for i in range(limit):
        res.append(Exp('from_int', [i]))
    
    local_exp = []
    for e in res:
        local_exp.append(Exp('absolute', [e]))
    for op in {'+', '-'}:
        for e1 in res:
            for e2 in res:
                local_exp.append(Exp('bin_op', [op, e1, e2]))
    for e in local_exp:
        res.append(e)
    return res



def enumerate_bexpressions(max_offset=5, seed_expressions=[], seed_positions=[]) -> list[BExp]:
    res = []

    for op in {'eq', 'lt', 'gt'}:
        for e1 in seed_expressions:
            for e2 in seed_expressions:
                res.append(BExp('bin_op', [op, e1, e2]))

    local_bexp = []
    #for op in {'and', 'or', }:
    #    for b1 in res:
    #        for b2 in res:
    #            local_bexp.append(BExp('bin_op', [op, b1, b2]))
    #
    for offset in range(-1*max_offset, max_offset):
        for pos in seed_positions:
            for prop in Prop:
                local_bexp.append(BExp('check_prop', [prop, pos, offset]))
    for bexp in local_bexp:
        res.append(bexp)
    return res