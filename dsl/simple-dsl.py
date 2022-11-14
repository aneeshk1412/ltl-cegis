#!/usr/bin/python3

from itertools import product, count

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

rules = {
    'POS': [
        ('next_goal_pos',),
        ('next_door_pos',),
        ('agent_pos',),
    ],
    'DIR': [
        ('Direction', 'POS', 'POS',),
        ('agent_dir',),
    ],
    'OBJ': [
        ('door',),
        ('wall',),
        ('goal',),
    ],
    'ABEXP': [
        ('InDirection', 'DIR', 'DIR',),
        ('InFrontOfAgent', 'OBJ',),
    ],
    'CMBEXP': [
        ('ABEXP',),
        ('Not', 'ABEXP',),
        ('And', 'CMBEXP', 'CMBEXP',),
        ('Or', 'CMBEXP', 'CMBEXP',),
    ],
    'BEXP': [
        ('true',),
        ('false',),
        ('CMBEXP',),
    ]
}


def productions(atom, depth):
    if depth == 1 and atom not in rules:
        yield atom
    if depth > 1:
        if atom not in rules:
            yield atom
        else:
            for rule in rules[atom]:
                for tup in iproduct(*[productions(term, depth-1) for term in rule]):
                    yield tup

start = 'BEXP'
depth = 8
for x in productions(start, depth):
    pass
