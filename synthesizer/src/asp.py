#!/usr/bin/python3
__author__ = "Kia Rahmani"


from random import seed
import string
from enum import Enum
from typing import List
from src.help import bcolors
from src.constants import _NUMBER_OF_AGENTS, _HALLWAY_LENGTH


class Prop(Enum):
    WALL = 0
    GOAL = 1

    def __str__(self):
        # return  bcolors.HEADER + str(self.name).lower() + bcolors.ENDC
        return str(self.name).lower()


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    NONE = 2

    def __str__(self):
        return bcolors.HEADER + str(self.name) + bcolors.ENDC


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


class ASP:
    def __init__(self, id, transition_cond_pairs, else_action) -> None:
        self.transition_cond_pairs = transition_cond_pairs
        self.id = id
        self.else_action = else_action

    def pretty_str(self) -> string:
        res = 'action_selection_policy_'+str(self.id) + " {\n"
        if_stmt = '    IF '
        for (b, a) in self.transition_cond_pairs:
            res += if_stmt+'(' + b.pretty_str() + \
                ') TAKE ' + str(a).replace('Action.', '') + ';\n'
            if_stmt = '    ELIF '
        res += "    ELSE TAKE " + \
            str(self.else_action).replace('Action.', '') + ';\n'
        res += "}"
        return res

