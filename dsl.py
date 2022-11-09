#!/usr/bin/python

from utils import *


class Action(Terminal):
    __match_args__ = ('term',)
    mapping = {'NONE': -1}

    def __init__(self, term) -> None:
        super().__init__(term)
        self.value = self.mapping[self.term]

    def __verify__(self) -> bool:
        if self.term in self.mapping.keys():
            return True
        return False

    @classmethod
    def __simple_enumerate__(cls):
        yield from [cls(k) for k in cls.mapping.keys()]


class StaticProperty(Terminal):
    props = set()

    def __verify__(self) -> bool:
        if self.term in self.props:
            return True
        return False

    @classmethod
    def __simple_enumerate__(cls):
        yield from [cls(p) for p in cls.props]

    @classmethod
    def __param_enumerate_1__(cls, props_list):
        yield from [cls(x) for x in props_list if x in cls.props]


class Vector(Terminal):
    finite_iter = None

    def __verify__(self) -> bool:
        return True
        # XXX
        # TODO: this needs to be fixed
        match self.term:
            case -3 | -2 | -1 | 1 | 2 | 3:
                return True
        return False

    @classmethod
    def __simple_enumerate__(cls):
        if not cls.finite_iter:
            cls.finite_iter = [cls(x) for x in ['?']]
        yield from cls.finite_iter

    @classmethod
    def __param_enumerate_1__(cls, max_vision):
        L = [cls(x) for x in ['?']]
        yield from L


def get_terminals_from_config(config):
    Action.mapping.update(dict((act['name'], act['value'])
                          for act in config['Action']))
    StaticProperty.props = set(prop['name']
                               for prop in config['StaticProperty'])

# <TODO> handling vector add quickly in 2D


class Position(NonTerminal):
    finite_iter = None

    def __verify__(self) -> bool:
        match self.terms:
            case ['StateRobotPos']:
                return True
            case ['vector_add', 'StateRobotPos', Vector()]:
                return True
        return False

    def __pstr__(self) -> str:
        return self.__cstr__()

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
            cls.finite_iter = [cls('StateRobotPos')] + [cls('vector_add',
                                                            'StateRobotPos', v) for v in Vector.__simple_enumerate__()]
        yield from cls.finite_iter

    @classmethod
    def __param_enumerate_1__(cls, max_vision):
        L = [cls('StateRobotPos')] + [cls('vector_add', 'StateRobotPos', v)
                                      for v in Vector.__param_enumerate_1__(max_vision)]
        yield from L

    def eval(self, state):
        match self.terms:
            case ['StateRobotPos']:
                return state['StateRobotPos']
            case ['vector_add', 'StateRobotPos', Vector()]:
                return state['StateRobotPos'] + self.terms[2].term


# <TODO> parse checkprop from description file to make the function
def check_prop(pos, static_prop):
    if static_prop.term == 'WALL':
        # XXX THIS MUST NOT BE HARD CODED
        return not (pos >= -500 and pos <= 500)
    elif static_prop.term == 'CHECKPOINT':
        return pos == 0
    elif static_prop.term == 'WINDOW':
        return pos in range(0, 5)
    else:
        raise NotImplementedError

# <TODO> separate Boolexp from Atomic Boolean Exp


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

    def __pstr__(self) -> str:
        return self.__str__()

    def __cstr__(self) -> str:
        match self.terms:
            case ['True']:
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
        # yield from (cls('eq', 'StateRobotAct', a) for a in Action.__simple_enumerate__())
        # yield from (cls('not', cls('eq', 'StateRobotAct', a)) for a in Action.__simple_enumerate__())
        yield from (cls('check_prop', pos, prop) for pos in Position.__simple_enumerate__() for prop in StaticProperty.__simple_enumerate__())
        yield from (cls('not', cls('check_prop', pos, prop)) for pos in Position.__simple_enumerate__() for prop in StaticProperty.__simple_enumerate__())
        for x, y in icombinations(cls.__simple_enumerate__):
            yield cls('and', x, y)
            yield cls('or', x, y)

    @classmethod
    def __param_enumerate_1__(cls, max_vision, props_list):
        yield from [cls('True'), cls('False')]
        # yield from (cls('eq', 'StateRobotAct', a) for a in Action.__simple_enumerate__())
        # yield from (cls('not', cls('eq', 'StateRobotAct', a)) for a in Action.__simple_enumerate__())
        yield from (cls('check_prop', pos, prop) for pos in Position.__param_enumerate_1__(max_vision) for prop in StaticProperty.__param_enumerate_1__(props_list))
        yield from (cls('not', cls('check_prop', pos, prop)) for pos in Position.__param_enumerate_1__(max_vision) for prop in StaticProperty.__param_enumerate_1__(props_list))
        def gen(): return cls.__param_enumerate_1__(max_vision, props_list)
        for x, y in icombinations(gen):
            yield cls('and', x, y)
            yield cls('or', x, y)

    def eval(self, state):
        match self.terms:
            case ['True']:
                return True
            case ['False']:
                return False
            case ['eq', 'StateRobotAct', Action()]:
                return state['StateRobotAct'] == self.terms[2].value
            case ['check_prop', Position(), StaticProperty()]:
                return check_prop(self.terms[1].eval(state), self.terms[2])
            case ['and', BooleanExp(), BooleanExp()]:
                return self.terms[1].eval(state) and self.terms[2].eval(state)
            case ['or', BooleanExp(), BooleanExp()]:
                return self.terms[1].eval(state) or self.terms[2].eval(state)
            case ['not', BooleanExp()]:
                return not self.terms[1].eval(state)


class Transition(NonTerminal):
    if_else_list = [['if (', ') return '],
                    ['if (', ') return '],
                    ['else return ']]
    sep = ';\n'

    def __verify__(self) -> bool:
        match self.terms:
            case [[BooleanExp(), Action('LEFT')],
                  [BooleanExp(), Action('RIGHT')],
                  [Action('NONE')]]:
                return True
        return False

    def __str__(self) -> str:
        return f"{self.sep.join(''.join(str(ele) for ele in interleave(*iters)) for iters in zip(self.if_else_list, self.terms))}{self.sep}"

    def __cstr__(self) -> str:
        return f"{self.sep.join(''.join(cstr(ele) for ele in interleave(*iters)) for iters in zip(self.if_else_list, self.terms))}{self.sep}"

    def __pstr__(self) -> str:
        return self.__cstr__()

    @classmethod
    def __simple_enumerate__(cls):
        for bexp1, bexp2 in iproduct(BooleanExp.__simple_enumerate__, BooleanExp.__simple_enumerate__):
            yield cls([bexp1, Action('LEFT')],
                      [bexp2, Action('RIGHT')],
                      [Action('NONE')])

    @classmethod
    def __param_enumerate_1__(cls, max_vision, props_list):
        def gen(): return BooleanExp.__param_enumerate_1__(max_vision, props_list)
        for bexp1, bexp2 in iproduct(gen, gen):
            yield cls([bexp1, Action('LEFT')],
                      [bexp2, Action('RIGHT')],
                      [Action('NONE')])

    def eval(self, state):
        for term in self.terms[:-1]:
            if term[0].eval(state):
                return term[1].value
        return self.terms[-1][0].value


class ASP(NonTerminal):
    def __verify__(self) -> bool:
        match self.terms:
            case [[Action('LEFT'), Transition()],
                  [Action('RIGHT'), Transition()],
                  [Action('NONE')]]:
                return True
        return False

    def __str__(self) -> str:
        res = ''
        for act, transition in self.terms[:-1]:
            res += f"if ({str(BooleanExp('eq', 'StateRobotAct', act))}) "
            res += '{\n'
            res += str(transition)
            res += '}\n'
        res += f'return {str(self.terms[-1][0])};\n'
        return res

    def __pstr__(self) -> str:
        res = ''
        for act, transition in self.terms[:-1]:
            for x, y in transition.terms[:-1]:
                local = str(act).ljust(5,' ') + ' -> ' + str(y) 
                local = local.ljust(15, ' ')
                local += ": " + x.__pstr__()
                res += local + '\n'
        res += '*     -> NONE  : Else'
        return res

    def __cstr__(self) -> str:
        res = ''
        for act, transition in self.terms[:-1]:
            res += f"if ({cstr(BooleanExp('eq', 'StateRobotAct', act))}) "
            res += '{\n'
            res += cstr(transition)
            res += '}\n'
        res += f'return {str(self.terms[-1][0])};\n'
        return res

    @classmethod
    def __simple_enumerate__(cls):
        for tran1, tran2 in iproduct(Transition.__simple_enumerate__, Transition.__simple_enumerate__):
            yield cls([Action('LEFT'), tran1],
                      [Action('RIGHT'), tran2],
                      [Action('NONE')])

    @classmethod
    def __param_enumerate_1__(cls, max_vision, props_list):
        def gen(): return Transition.__param_enumerate_1__(max_vision, props_list)
        for tran1, tran2 in iproduct(gen, gen):
            yield cls([Action('LEFT'), tran1],
                      [Action('RIGHT'), tran2],
                      [Action('NONE')])

    def eval(self, state):
        for act, transition in self.terms[:-1]:
            if state['StateRobotAct'] == act:
                return transition.eval(state)
        return self.terms[-1][0].value


if __name__ == '__main__':
    config = open_config_file('descriptions/1d-hallway.yml')
    get_terminals_from_config(config)

    wall = StaticProperty('WALL')
    left_act = Action('LEFT')
    right_act = Action('RIGHT')
    dist1 = Vector('?')
    dist2 = Vector('?')
    pos1 = Position('vector_add', 'StateRobotPos', dist1)
    pos2 = Position('vector_add', 'StateRobotPos', dist2)

    bexpL = BooleanExp('check_prop', pos1, wall)
    bexpR = BooleanExp('check_prop', pos2, wall)
    tranL = Transition([BooleanExp('not', bexpL), Action('LEFT')], [
                       bexpL, Action('RIGHT')], [Action('NONE')])
    tranR = Transition([bexpR, Action('LEFT')], [BooleanExp(
        'True'), Action('RIGHT')], [Action('NONE')])

    asp = ASP([left_act, tranL], [right_act, tranR], [Action('NONE')])

    print(asp, end='\n\n')
    print(cstr(asp), end='\n\n')

    # print(bexp1, end='\n\n')
    # print(bexp2, end='\n\n')
    # print(asp, end='\n\n')

    # print(cstr(bexp1), end='\n\n')
    # print(cstr(bexp2), end='\n\n')
    # print(cstr(asp), end='\n\n')

    i = 0
    for x in ASP.__simple_enumerate__():
        i += 1
        # print(cstr(x), end='\n\n')
        if cstr(x) == cstr(asp):
            print(f'Found Target Program at Iteration: {i}')
            print(cstr(x))
            break

    i = 0
    for x in ASP.__param_enumerate_1__(max_vision=1, props_list=['WALL']):
        i += 1
        # print(cstr(x), end='\n\n')
        if cstr(x) == cstr(asp):
            print(f'Found Target Program at Iteration: {i}')
            print(x)
            break

'''
    if ((StateRobotAct == RIGHT && check_prop_WALL(vector_add(StateRobotPos, 1)))) return LEFT;
    if ((StateRobotAct == LEFT && check_prop_WALL(vector_add(StateRobotPos, -1)))) return RIGHT;
'''
