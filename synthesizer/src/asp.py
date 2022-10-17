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
        # return  bcolors.HEADER + str(self.name).lower() + bcolors.ENDC
        return str(self.name).lower()


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    NONE = 2

    def __str__(self):
        return bcolors.HEADER + str(self.name) + bcolors.ENDC


class ASP:
    def __init__(self, id, transition_cond_pairs, else_action) -> None:
        self.transition_cond_pairs = transition_cond_pairs
        self.id = id
        self.else_action = else_action

    def pretty_str(self) -> String:
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

