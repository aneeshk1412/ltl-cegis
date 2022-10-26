#!/usr/bin/python3

from coder import make_model_program
from dsl import ASP
from utils import cstr
from verifier.verify import verifies

def algorithm_1():
    for prog in ASP.__simple_enumerate__():
        b, trace = verifies(cstr(prog), get_counterexample=True)
        print(prog, end='\n')
        print(b, trace)
        print()

        if b:
            print("SAT Prog:")
            print(prog)
            break

if __name__ == '__main__':
    with open('verifier/model_prog.c', 'w') as f:
        f.write(str(make_model_program())) 
    algorithm_1()