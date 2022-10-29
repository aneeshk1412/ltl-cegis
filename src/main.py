#!/usr/bin/python3

from coder import make_model_program
from dsl import ASP
from utils import cstr, grouped2
from verifier.verify import verifies
import argparse

def algorithm_1():
    i = 0
    for prog in ASP.__simple_enumerate__():
        i += 1
        b, trace = verifies(cstr(prog), get_counterexample=True)
        print(prog, end='\n')
        print(b)
        print()

        if b:
            print(f"SAT Prog {i} :")
            print(prog)
            break

def algorithm_2():
    negative_demo_set = list()
    positive_demo_set = list()
    i = 0
    for prog in ASP.__simple_enumerate__():
        i += 1
        b = False
        flag = False
        print(prog, end='\n')
        for demo in negative_demo_set:
            if all(prog.eval(s) == a for s, a in demo):
                print(b, 'unsat demo')
                print()
                # print(negative_demo_set, end='\n\n'+150*'-'+'\n')
                flag = True
                break
        
        if flag:
            continue

        b, trace = verifies(cstr(prog), get_counterexample=True)
        print(b)
        print()

        if b:
            print(f"SAT Prog {i}:")
            print(prog)
            break

        ## Parse Trace, add to negative demo
        demo = tuple(grouped2(trace))
        if demo not in negative_demo_set:
            negative_demo_set.append(demo)
        
        # print(negative_demo_set, end='\n\n'+150*'-'+'\n')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    with open('verifier/model_prog.c', 'w') as f:
        f.write(str(make_model_program())) 
    args = get_args()
    if args.alg == 1:
        algorithm_1()
    elif args.alg == 2:
        algorithm_2()