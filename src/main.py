#!/usr/bin/python3

from os import popen
from coder import make_model_program
from dsl import *
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
        print('-'*75)
        i += 1
        b = False
        flag = False
        print('asp_'+str(i))
        print(prog, end='\n')
        for demo in negative_demo_set:
            if all(prog.eval(s) == a for s, a in demo):
                print(b, '(due to unsat demo)')
                print()
                # print(negative_demo_set, end='\n\n'+150*'-'+'\n')
                flag = True
                break

        if flag:
            continue  # this will save us a call to the ultimate because we already know the candidate ASP matches one of the negative demos

        # TODO: make the spec to be verified an arg to the "verifies" function
        b, trace = verifies(cstr(prog), get_counterexample=True)
        print(b)
        print()

        if b:
            print()
            print('#'*100)
            print(f"SAT Prog {i}:")  # an ASP consistent with the spec is found
            print(prog)
            break

        # the ASP violates the spec
        # Parse Trace, add to negative demo
        demo = tuple(grouped2(trace))
        if demo not in negative_demo_set:
            negative_demo_set.append(demo)

        # print(negative_demo_set, end='\n\n'+150*'-'+'\n')


# define a function to generate and return the ground truth
def gen_ground_truth():
    wall = StaticProperty('WALL')
    pos1 = Position('vector_add', 'StateRobotPos', Vector(10))
    pos2 = Position('vector_add', 'StateRobotPos', Vector(-10))

    bexp1 = BooleanExp('check_prop', pos1, wall)
    bexp2 = BooleanExp('check_prop', pos2, wall)
    asp = ASP([bexp1, Action('LEFT')], 
              [bexp2, Action('RIGHT')], 
              [Action('RIGHT')])
    return asp


def compute_max_vision(action_samples, action: int):
    return 20


def compute_props_list(action_samples, action: int):
    return ['WALL']


def algorithm_3():
    _right = 101
    _left = 100
    #ground_truth = gen_ground_truth()
    # states under which the action was NOT taken
    left_action_negative_samples  = [#{'StateRobotAct': _left,  'StateRobotPos': 0},
                                     #{'StateRobotAct': _right, 'StateRobotPos': -100},
                                     #{'StateRobotAct': _left,  'StateRobotPos': 485},
                                     #{'StateRobotAct': _left,  'StateRobotPos': -495}
                                    ]

    right_action_negative_samples = [#{'StateRobotAct': _right, 'StateRobotPos': 495}
                                    ]
    # states under which the action was taken
    left_action_positive_samples  = [#{'StateRobotAct': _right, 'StateRobotPos': 491}
                                    ]

    right_action_positive_samples = [#{'StateRobotAct': _left,  'StateRobotPos': 0},
                                     #{'StateRobotAct': _right, 'StateRobotPos': 1},
                                     #{'StateRobotAct': _right, 'StateRobotPos': 100},
                                     #{'StateRobotAct': _right, 'StateRobotPos': 485},
                                     #{'StateRobotAct': _right, 'StateRobotPos': -499}
                                    ]

    action_samples = {_left:  {'+': left_action_positive_samples,
                               '-': left_action_negative_samples},
                      _right: {'+': right_action_positive_samples,
                               '-': right_action_negative_samples}}

    # TODO
    max_vision = compute_max_vision(action_samples, 0)
    props_list = compute_props_list(action_samples, 0)

    i = 0
    # TODO: make the input args specific for each type of action, e.g. a max_vision for right, and another one for left
    for prog in ASP.__param_enumerate_1__(max_vision=max_vision, props_list=props_list):
        print('-'*75)
        i += 1
        b = False
        flag = False
        print('asp_'+str(i))
        print('   '+str(prog).replace('\n','\n   '))
        # check samples
        for a in {_right, _left}:
            if not all(prog.eval(p) == a for p in action_samples[a]['+']) or not all(prog.eval(p) != a for p in action_samples[a]['-']):
                print('discarded: candidate program is not consistent with demos')
                flag = True
                break
        # this will save us a call to the ultimate because we already know the candidate ASP matches one of the negative demos
        if flag:
            continue

        input('ok?')

        b, trace = verifies(cstr(prog), get_counterexample=True)

        if b:  # found a consistent program with the spec
            print('#'*100)
            print('found a program consistent with the specification and demonstrations!')
            print(f"SAT Prog {i}:")  # an ASP consistent with the spec is found
            print('     '+prog.__str__().replace('\n', '\n     '))
            break
        else:  # the ASP violates the spec
            demo = tuple(grouped2(trace))
            for s, a in demo:
                action_samples[a]['-'].append(s)

#


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
    else:
        print('running algorithm 3')
        algorithm_3()
