#!/usr/bin/python3
__author__ = "Kia Rahmani"


from random import seed
from tokenize import String
from enum import Enum
from typing import List
from src.help import bcolors
from src.constants import _NUMBER_OF_AGENTS, _HALLWAY_LENGTH


class Prop(Enum):
    WALL = 0
    GOAL = 1
    def __str__(self):
        #return  bcolors.HEADER + str(self.name).lower() + bcolors.ENDC
        return str(self.name).lower() 



class Action(Enum):
    LEFT = 0
    RIGHT = 1
    NONE = 2
    def __str__(self):
         return  bcolors.HEADER + str(self.name) + bcolors.ENDC




class ASP:
    def __init__(self, id, else_action) -> None:
        self.transition_cond_pairs = list()
        self.id = id
        self.else_action = else_action

    def add_transition_cond_pair(self, b, a):
        self.transition_cond_pairs.append((b, a))

    def pretty_str(self) -> String:
        res = 'action_selection_policy_'+str(self.id) + " {\n"
        if_stmt = '    IF '
        for (b, a) in self.transition_cond_pairs:
            res += if_stmt+'(' + b.pretty_str() + \
                ') TAKE ' + str(a).replace('Action.','') + ';\n'
            if_stmt = '    ELIF '
        res += "    ELSE TAKE " + str(self.else_action).replace('Action.','')  + ';\n'
        res += "}"
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
        assert (tp in {'from_pos', 'from_int', 'distance'})
        if tp == 'distance':
            assert (len(children) == 2)
        else:
            assert (len(children) == 1)
        self.tp = tp
        self.children = children

    def pretty_str(self) -> String:
        if self.tp == 'from_pos':
            return self.children[0].pretty_str()
        elif self.tp == 'from_int':
            return 'Exp('+str(self.children[0])+')'
        elif self.tp == 'distance':
            return 'dist(' + self.children[0].pretty_str() + ',' + \
                self.children[1].pretty_str() + ')'
        else:
            raise Exception("unexpected expression type")


class BExp:
    def __init__(self, tp, children) -> None:
        #assert (tp in {'check_prop', 'bin_op', 'check_robot_action'})
        assert (len(children) == 3)
        if tp == 'bin_op':
            assert (children[0] in {'and', 'or', 'eq', 'lt', 'gt'})
        self.tp = tp
        self.children = children

    def pretty_str(self) -> String:
        res = ''
        if self.tp == 'bin_op':
            res = self.children[0] + '(' + self.children[1].pretty_str() + ',' + self.children[2].pretty_str() + ')'
        elif self.tp == 'check_prop':
            res = 'check_' + str(self.children[0]).replace('Prop.', '').lower() + \
                '(at=' + self.children[1].pretty_str() + \
                ', offset=' + str(self.children[2]) + ')'
        else:
            raise Exception("unexpected boolean expression type")
        return bcolors.FAIL + res + bcolors.ENDC
        #return res

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

    local_exp = []
    for e1 in res:
        for e2 in res:
            # optimization: dist of the same position does not make sense
            if e1 == e2:
                continue
            # optimization: dist of constant positions does not make sense
            if e1.children[0].tp == 'from_int' and e2.children[0].tp == 'from_int':
                continue
            local_exp.append(Exp('distance', [e1, e2]))
    for e in local_exp:
        res.append(e)
    
    for i in range(limit):
        res.append(Exp('from_int', [i]))
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
     #       for b2 in res:
     #           local_bexp.append(BExp('bin_op', [op, b1, b2]))
    #
    for offset in range(-1*max_offset, max_offset):
        for pos in seed_positions:
            for prop in Prop:
                local_bexp.append(BExp('check_prop', [prop, pos, offset]))
    for bexp in local_bexp:
        res.append(bexp)
    return res
