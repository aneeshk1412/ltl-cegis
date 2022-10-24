#!/usr/bin/python

from utils import *

class Action(Terminal):
    __match_args__ = ('term',)
    finite_iter = None
    def __verify__(self) -> bool:
        match self.term:
            case 'LEFT' | 'RIGHT':
                return True
        return False

    @classmethod
    def __simple_enumerate__(cls):
        if not cls.finite_iter:
            cls.finite_iter = [cls(a) for a in ['LEFT', 'RIGHT']]
        yield from cls.finite_iter


class StaticProperty(Terminal):
    finite_iter = None
    def __verify__(self) -> bool:
        match self.term:
            case 'WALL':
                return True
        return False

    @classmethod
    def __simple_enumerate__(cls):
        if not cls.finite_iter:
            cls.finite_iter = [cls(p) for p in ['WALL']]
        yield from cls.finite_iter


class Vector(Terminal):
    finite_iter = None
    def __verify__(self) -> bool:
        match self.term:
            case -3 | -2 | -1 | 1 | 2 | 3:
                return True
        return False

    @classmethod
    def __simple_enumerate__(cls):
        if not cls.finite_iter:
            cls.finite_iter = [cls(x) for x in [-3 , -2 , -1 , 1 , 2 , 3]]
        yield from cls.finite_iter


class Position(NonTerminal):
    finite_iter = None
    def __verify__(self) -> bool:
        match self.terms:
            case ['StateRobotPos']:
                return True
            case ['vector_add', 'StateRobotPos', Vector()]:
                return True
        return False
    
    def __str__(self) -> str:
        match self.terms:
            case ['StateRobotPos']:
                return 'StateRobotPos'
            case ['vector_add', 'StateRobotPos', Vector()]:
                return f'{self.terms[0]}({self.terms[1]}, {str(self.terms[2])})'
    
    def __cstr__(self) -> str:
        match self.terms:
            case ['StateRobotPos']:
                return 'StateRobotPos'
            case ['vector_add', 'StateRobotPos', Vector()]:
                return f'{self.terms[0]}({self.terms[1]}, {cstr(self.terms[2])})'

    @classmethod
    def __simple_enumerate__(cls):
        if not cls.finite_iter:
            cls.finite_iter = [cls('StateRobotPos')] + [cls('vector_add', 'StateRobotPos', v) for v in Vector.__simple_enumerate__()]
        yield from cls.finite_iter


class BooleanExp(NonTerminal):
    operation_map = {'and': '&&', 'or': '||', 'not': '!'}
    def __verify__(self) -> bool:
        match self.terms:
            case ['True' | 'False']:
                return True
            case ['eq', 'StateRobotAct', Action()]:
                return True
            case ['not', BooleanExp()]:
                return True
            case ['check_prop', Position(), StaticProperty()]:
                return True
            case ['and' | 'or', BooleanExp(), BooleanExp()]:
                return True
        return False

    def __str__(self) -> str:
        match self.terms:
            case ['True' | 'False']:
                return str(self.terms[0])
            case ['not', BooleanExp()]:
                return f'{self.terms[0]}({str(self.terms[1])})'
        return f'{self.terms[0]}({str(self.terms[1])}, {str(self.terms[2])})'
    
    def __cstr__(self) -> str:
        match self.terms:
            case ['True' ]:
                return '1'
            case ['False']:
                return '0'
            case ['eq', 'StateRobotAct', Action()]:
                return f'{self.terms[1]} == {self.terms[2]}'
            case ['check_prop', Position(), StaticProperty()]:
                return f'{self.terms[0]}_{self.terms[2]}({cstr(self.terms[1])})'
            case ['and' | 'or', BooleanExp(), BooleanExp()]:
                return f'({cstr(self.terms[1])} {self.operation_map[self.terms[0]]} {cstr(self.terms[2])})'
            case ['not', BooleanExp()]:
                return f'{self.operation_map[self.terms[0]]}({cstr(self.terms[1])})'

    @classmethod
    def __simple_enumerate__(cls):
        yield from [cls('True'), cls('False')]
        yield from (cls('eq', 'StateRobotAct', a) for a in Action.__simple_enumerate__())
        yield from (cls('not', cls('eq', 'StateRobotAct', a)) for a in Action.__simple_enumerate__())
        yield from (cls('check_prop', pos, prop) for pos in Position.__simple_enumerate__() for prop in StaticProperty.__simple_enumerate__())
        yield from (cls('not', cls('check_prop', pos, prop)) for pos in Position.__simple_enumerate__() for prop in StaticProperty.__simple_enumerate__())
        for x, y in icombinations(cls.__simple_enumerate__):
            yield cls('and', x, y)
            yield cls('or', x, y)


class ASP(NonTerminal):
    if_else_list = [['if (', ') return '], 
                    ['if (', ') return ']]
    sep = ';\n'
 
    def __verify__(self) -> bool:
        match self.terms:
            case [[BooleanExp(), Action('LEFT')], 
                  [BooleanExp(), Action('RIGHT')]]:
                return True
        return False

    def __str__(self) -> str:
        return f"{self.sep.join(''.join(str(ele) for ele in interleave(*iters)) for iters in zip(self.if_else_list, self.terms))}{self.sep}"

    def __cstr__(self) -> str:
        return f"{self.sep.join(''.join(cstr(ele) for ele in interleave(*iters)) for iters in zip(self.if_else_list, self.terms))}{self.sep}"
    
    @classmethod
    def __simple_enumerate__(cls):
        for bexp1, bexp2 in iproduct(BooleanExp.__simple_enumerate__, BooleanExp.__simple_enumerate__):
            yield cls([bexp1, Action('LEFT')], 
                      [bexp2, Action('RIGHT')])


if __name__ == '__main__':
    wall = StaticProperty('WALL')
    left_act = Action('LEFT')
    right_act = Action('RIGHT')
    dist1 = Vector(1)
    dist2 = Vector(-1)
    pos1 = Position('vector_add', 'StateRobotPos', dist1)
    pos2 = Position('vector_add', 'StateRobotPos', dist2)

    bexp1 = BooleanExp('and', BooleanExp('eq', 'StateRobotAct', right_act), BooleanExp('check_prop', pos1, wall))
    bexp2 = BooleanExp('and', BooleanExp('eq', 'StateRobotAct', left_act), BooleanExp('check_prop', pos2, wall))
    asp = ASP([bexp1, Action('LEFT')], [bexp2, Action('RIGHT')])

    print(bexp1, end='\n\n')
    print(bexp2, end='\n\n')
    print(asp, end='\n\n')

    print(cstr(bexp1), end='\n\n')
    print(cstr(bexp2), end='\n\n')
    print(cstr(asp), end='\n\n')

    i = 0
    for x in ASP.__simple_enumerate__():
        i += 1
        print(x, end='\n\n')
        if cstr(x) == cstr(asp):
            print(f'Found Target Program at Iteration: {i}')
            break