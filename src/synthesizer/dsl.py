#!/usr/bin/python3
__coauthors__ = ["Kia Rahmani", "Aneesh Shetty"]

import string
from enum import Enum
from .help import bcolors

def cstr(obj):
    try:
        return obj.__cstr__()
    except AttributeError:
        return obj.__str__()

class Prop(Enum):
    WALL = 103
    GOAL = 104

    def __str__(self):
        return self.name.lower()
    
    def __cstr__(self):
        return self.name

class Action(Enum):
    LEFT = 100
    RIGHT = 101
    NONE = 102

    def __str__(self):
        return bcolors.HEADER + self.name + bcolors.ENDC
    
    def __cstr__(self):
        return self.name


class Position:
    def __init__(self, tp: string, children):
        assert (tp in {'robot_pos', 'agent_pos'})
        if (tp == 'robot_pos'):
            (len(children) == 0)
        else:
            assert (len(children) == 1)
        self.tp = tp
        self.children = children

    def __str__(self) -> string:
        if self.tp == 'robot_pos':
            return 'R#Pos'
        elif self.tp == 'agent_pos':
            return 'A'+str(self.children[0])+'#Pos'
        else:
            raise Exception("unexpected position type")
    
    def __cstr__(self) -> string:
        if self.tp == 'robot_pos':
            return 'StateRobotPos'
        elif self.tp == 'agent_pos':
            return 'AgentPos'+ cstr(self.children[0])
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

    def __str__(self) -> string:
        if self.tp == 'from_pos':
            return str(self.children[0])
        elif self.tp == 'from_int':
            return ''+str(self.children[0])+''
        elif self.tp == 'diff':
            return 'diff(' + str(self.children[0]) + ',' + \
                str(self.children[1]) + ')'
        else:
            raise Exception("unexpected expression type")
    
    def __cstr__(self) -> string:
        if self.tp == 'from_pos':
            return cstr(self.children[0])
        elif self.tp == 'from_int':
            return cstr(self.children[0])
        elif self.tp == 'diff':
            return 'diff(' + cstr(self.children[0]) + ',' + \
                cstr(self.children[1]) + ')'
        else:
            raise Exception("unexpected expression type")


class AtomicBoolExp:
    op_map = {'lt': '<', 'gt': '>', 'eq': '=='}
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

    def __str__(self) -> string:
        res = ''
        if self.tp == 'from_bool':
            res = str(self.children[0])
        elif self.tp == 'curr_rob_act':
            res = 'current_action_is('+str(self.children[0])+')'
        elif self.tp in {'lt', 'gt', 'eq'}:
            res = self.tp + \
                "("+str(self.children[0]) + \
                ',' + str(self.children[1])+')'
        elif self.tp == 'check_prop':
            res = 'check_' + \
                str(self.children[1]) + '(' + str(self.children[0]) + \
                ',' + str(self.children[2]) + ')'
        else:
            raise Exception("unexpected boolean expression type")
        return bcolors.FAIL + res + bcolors.ENDC

    def __cstr__(self) -> string:
        res = ''
        if self.tp == 'from_bool':
            res = '1' if self.children[0] else '0'
        elif self.tp == 'curr_rob_act':
            res = '(StateRobotAct == '+cstr(self.children[0])+')'
        elif self.tp in {'lt', 'gt', 'eq'}:
            res = "("+cstr(self.children[0]) + self.op_map[self.tp] + \
                cstr(self.children[1])+')'
        elif self.tp == 'check_prop':
            res = 'check_' + \
                cstr(self.children[1]) + '(' + cstr(self.children[0]) + \
                ',' + cstr(self.children[2]) + ')'
        else:
            raise Exception("unexpected boolean expression type")
        return res


class BoolExp:
    def __init__(self, tp: string, children) -> None:
        assert (tp in {'and', 'or'})
        assert (len(children) == 2)
        self.tp = tp
        self.children = children

    def __str__(self) -> string:
        if self.tp == 'and':
            op = '∧'
        elif self.tp == 'or':
            op = '∨'
        return '(' + str(self.children[0]) + ' ' + op + ' ' + \
            str(self.children[1]) + ')'
    
    def __cstr__(self) -> string:
        if self.tp == 'and':
            op = '&&'
        elif self.tp == 'or':
            op = '||'
        return '(' + cstr(self.children[0]) + ' ' + op + ' ' + \
            cstr(self.children[1]) + ')'


class ASP:
    def __init__(self, id, transition_cond_pairs, else_action) -> None:
        self.transition_cond_pairs = transition_cond_pairs
        self.id = id
        self.else_action = else_action

    def get_predicate_for_action(self, action: Action) -> BoolExp:
        if action == self.else_action:
            raise Exception('not implemented yet')  # TODO
        else:
            for b, a in self.transition_cond_pairs:
                if a == action:
                    return b
            raise Exception(
                'the given asp does not define a predicate for the given action', a, action)

    def __str__(self) -> string:
        res = 'action_selection_policy_'+str(self.id) + " {\n"
        if_stmt = '    IF '
        for (b, a) in self.transition_cond_pairs:
            res += if_stmt+'(' + str(b) + \
                ') TAKE ' + str(a) + ';\n'
            if_stmt = '    IF '
        res += "    TAKE " + \
            str(self.else_action) + ';\n'
        res += "}"
        return res
    
    def __cstr__(self) -> string:
        res = ''
        if_stmt = '    if '
        for (b, a) in self.transition_cond_pairs:
            res += if_stmt+'(' + cstr(b) + \
                ') return ' + cstr(a) + ';\n'
            if_stmt = '    if '
        res += "    return " + \
            cstr(self.else_action) + ';\n'
        return res
