#!/usr/bin/python3

from abc import ABC, abstractmethod
from itertools import product, count, repeat

class GenCacher:
    def __init__(self, generator):
        self.g = generator
        self.cache = []

    def __getitem__(self, idx):
        while len(self.cache) <= idx:
            try:
                self.cache.append(next(self.g))
            except StopIteration:
                break
        try:
            return self.cache[idx]
        except IndexError:
            return None

def summations(sum_to, n=2):
    if n == 1:
        yield (sum_to,)
    else:
        for head in range(sum_to + 1):
            for tail in summations(sum_to - head, n - 1):
                yield (head,) + tail

def iproduct(*gens):
    gens = list(map(GenCacher, gens))
    num_gens = len(gens)

    for dist in count(0):
        no_tuples_found = True
        for idxs in summations(dist, num_gens):
            tup = tuple(gen[idx] for gen, idx in zip(gens, idxs))
            if any(t is None for t in tup):
                continue
            no_tuples_found = False
            yield tup
        if no_tuples_found:
            break

def cstr(obj):
    try:
        return obj.__cstr__()
    except AttributeError:
        return str(obj)

def interleave(*iterables):
    yield from (ele for tups in zip(*iterables) for ele in tups)

def icombinations(f):
    for i, x in zip(count(), f()):
        yield from zip(f(), repeat(x, i + 1))


class Terminal(ABC):
    def __init__(self, term) -> None:
        self.term = term
        if not self.__verify__():
            print(self.term)
            raise NotImplementedError

    def __str__(self) -> str:
        return str(self.term)

    def __cstr__(self) -> str:
        return str(self.term)

    @abstractmethod
    def __verify__(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def __simple_enumerate__(cls):
        pass


class NonTerminal(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.terms = list(args)
        if not self.__verify__():
            print(self.terms)
            raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        return ''

    @abstractmethod
    def __cstr__(self) -> str:
        return ''

    @abstractmethod
    def __verify__(self) -> bool:
        pass

    @classmethod
    @abstractmethod
    def __simple_enumerate__(cls):
        pass


class RobotAction(Terminal):
    __match_args__ = ('term',)
    mapping = dict()

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


class Vector(Terminal):
    max_robot_vision = None
    zero_vec = None
    finite_iter = None

    def __verify__(self) -> bool:
        if len(self.term) != len(self.max_robot_vision):
            return False
        if all(t == '?' for t in self.term):
            return True
        if any(abs(t) > v for t, v in zip(self.term, self.max_robot_vision.values())):
            return False
        return True

    @classmethod
    def __simple_enumerate__(cls):
        yield cls(tuple('?' for _ in cls.max_robot_vision))
        # if not cls.finite_iter:
        #     # <TODO> change to iterate from small to big
        #     cls.finite_iter = [cls(tup) for tup in product(*[range(-v, v+1) for v in cls.max_robot_vision.values()]) if tup != cls.zero_vec]
        # yield from cls.finite_iter


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
                return ', '.join(f'StateRobotPos{a}' for a in Vector.max_robot_vision)
            case ['vector_add', 'StateRobotPos', Vector()]:
                # <TODO> make the sign pretty
                return ', '.join(f'StateRobotPos{a} + {b}' for a, b in zip(Vector.max_robot_vision, self.terms[2].term))

    @classmethod
    def __simple_enumerate__(cls):
        if not cls.finite_iter:
            cls.finite_iter = [cls('StateRobotPos')] + [cls('vector_add', 'StateRobotPos', v) for v in Vector.__simple_enumerate__()]
        yield from cls.finite_iter


class AtomicBooleanExp(NonTerminal):
    def __verify__(self) -> bool:
        match self.terms:
            case ['check_prop', Position(), StaticProperty()]:
                return True
        return False

    def __str__(self) -> str:
        return f'{self.terms[0]}({str(self.terms[1])}, {str(self.terms[2])})'

    def __cstr__(self) -> str:
        match self.terms:
            case ['check_prop', Position(), StaticProperty()]:
                return f'{self.terms[0]}_{self.terms[2]}({cstr(self.terms[1])})'

    @classmethod
    def __simple_enumerate__(cls):
        yield from (cls('check_prop', pos, prop) for pos in Position.__simple_enumerate__() for prop in StaticProperty.__simple_enumerate__())


class CombinationBooleanExp(NonTerminal):
    operation_map = {'and': '&&', 'or': '||', 'not': '!'}

    def __verify__(self) -> bool:
        match self.terms:
            case [AtomicBooleanExp()]:
                return True
            case ['not', AtomicBooleanExp()]:
                return True
            case ['and' | 'or', CombinationBooleanExp(), CombinationBooleanExp()]:
                return True
        return False

    def __str__(self) -> str:
        match self.terms:
            case [AtomicBooleanExp()]:
                return str(self.terms[0])
            case ['not', AtomicBooleanExp()]:
                return f'{self.terms[0]}({str(self.terms[1])})'
        return f'{self.terms[0]}({str(self.terms[1])}, {str(self.terms[2])})'

    def __cstr__(self) -> str:
        match self.terms:
            case [AtomicBooleanExp()]:
                return cstr(self.terms[0])
            case ['and' | 'or', BooleanExp(), BooleanExp()]:
                return f'({cstr(self.terms[1])} {self.operation_map[self.terms[0]]} {cstr(self.terms[2])})'
            case ['not', AtomicBooleanExp()]:
                return f'{self.operation_map[self.terms[0]]}({cstr(self.terms[1])})'

    @classmethod
    def __simple_enumerate__(cls):
        yield from (cls(atbexp) for atbexp in AtomicBooleanExp.__simple_enumerate__())
        yield from (cls('not', atbexp) for atbexp in AtomicBooleanExp.__simple_enumerate__())
        for x, y in icombinations(cls.__simple_enumerate__):
            # <TODO> do not combine things that result in something in the past, or that result in UNSAT
            yield cls('and', x, y)
            yield cls('or', x, y)


class BooleanExp(NonTerminal):
    def __verify__(self) -> bool:
        match self.terms:
            case ['True' | 'False' | '?']:
                return True
            case [CombinationBooleanExp()]:
                return True
        return False

    def __str__(self) -> str:
        match self.terms:
            case ['True' | 'False' | '?']:
                return str(self.terms[0])
        return f'{str(self.terms[0])}'

    def __cstr__(self) -> str:
        match self.terms:
            case ['True']:
                return '1'
            case ['False']:
                return '0'
            case ['?']:
                return '?'
            case [CombinationBooleanExp()]:
                return f'{cstr(self.terms[0])}'

    @classmethod
    def __simple_enumerate__(cls):
        yield cls('?')
        yield from (cls(cbexp) for cbexp in CombinationBooleanExp.__simple_enumerate__())


class TransitionASP(NonTerminal):
    def __verify__(self) -> bool:
        if not isinstance(self.terms[0], dict):
            return False
        for key in product(RobotAction.mapping, RobotAction.mapping):
            if key not in self.terms[0]:
                return False
            if not isinstance(self.terms[0][key], BooleanExp):
                return False
        return True

    def __str__(self) -> str:
        res = ''
        for key in product(RobotAction.mapping, RobotAction.mapping):
            res += f'{key[0]} -> {key[1]} : '
            res += f'{str(self.terms[0][key])}\n'
        return res

    def __cstr__(self) -> str:
        res = ''
        for key in product(RobotAction.mapping, RobotAction.mapping):
            res += f'if (StateRobotAct == {key[0]} && {cstr(self.terms[0][key])}) return {key[1]};\n'
        return res

    @classmethod
    def __simple_enumerate__(cls):
        enumerator_list = [BooleanExp.__simple_enumerate__() for _ in product(RobotAction.mapping, RobotAction.mapping)]
        for conds in iproduct(*enumerator_list):
            yield cls(dict(zip(product(RobotAction.mapping, RobotAction.mapping), conds)))


class ASP(NonTerminal):
    def __verify__(self) -> bool:
        if not isinstance(self.terms[0], dict):
            return False
        for key in RobotAction.mapping:
            if key not in self.terms[0]:
                return False
            if not isinstance(self.terms[0][key], BooleanExp):
                return False
        return True

    def __str__(self) -> str:
        res = ''
        for key in RobotAction.mapping:
            res += f'{str(self.terms[0][key])} -> {key}\n'
        return res

    def __cstr__(self) -> str:
        res = ''
        for key in RobotAction.mapping:
            res += f'if ({cstr(self.terms[0][key])}) return {key};\n'
        return res

    @classmethod
    def __simple_enumerate__(cls):
        enumerator_list = [BooleanExp.__simple_enumerate__() for _ in RobotAction.mapping]
        for conds in iproduct(*enumerator_list):
            yield cls(dict(zip(RobotAction.mapping, conds)))


def get_terminals_from_config(config):
    RobotAction.mapping = dict((action, details['value']) for action, details in config['actionTypes']['robotAction']['actions'].items())
    StaticProperty.props = set(config['staticProperty'].keys())
    Vector.max_robot_vision = config['maxRobotVision']
    Vector.zero_vec = tuple(0 for _ in Vector.max_robot_vision)


def print_testing():
    import yaml
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, help='Filename')
    args = parser.parse_args()
    with open(args.f, 'r') as stream:
        config = yaml.safe_load(stream)
    get_terminals_from_config(config)

    # for act in RobotAction.__simple_enumerate__():
    #     print(act)
    #     print(cstr(act))
    # print()

    # for prop in StaticProperty.__simple_enumerate__():
    #     print(prop)
    #     print(cstr(prop))
    # print()

    # for tup in Vector.__simple_enumerate__():
    #     print(tup)
    #     print(cstr(tup))
    # print()

    # for tup in Position.__simple_enumerate__():
    #     print(tup)
    #     print(cstr(tup))
    # print()

    # for tup in AtomicBooleanExp.__simple_enumerate__():
    #     print(tup)
    #     print(cstr(tup))
    # print()

    # for tup in BooleanExp.__simple_enumerate__():
    #     print(tup)
    #     # print(cstr(tup))
    # print()

    for tup in TransitionASP.__simple_enumerate__():
        print(tup)
        # print(cstr(tup))
    print()

    # for tup in ASP.__simple_enumerate__():
    #     print(tup)
    #     # print(cstr(tup))
    # print()



if __name__ == '__main__':
    print_testing()
