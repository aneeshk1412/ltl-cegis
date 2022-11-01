#!/usr/bin/python3

from operator import neg
from coder import make_model_program
from dsl import *
from utils import cstr, grouped2, open_config_file
from verifier.verify import verifies
import argparse


def algorithm_1():
    i = 0
    for prog in ASP.__simple_enumerate__():
        i += 1
        b, _ = verifies(cstr(prog), get_counterexample=True)
        print(prog, end='\n')
        print(b)
        print()

        if b:
            print(f"SAT Prog {i} :")
            print(prog)
            break


def algorithm_2():
    negative_demo_set = list()
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
    def is_wall(index):
        return index <= -500 or index >= 500
    _max_hallway = 500
    positive_positions = list(map (lambda s: s['StateRobotPos'], action_samples[action]['+']))
    negative_positions = list(map (lambda s: s['StateRobotPos'], action_samples[action]['-']))
    result = _max_hallway
        

    
    for i in reversed(range(_max_hallway+1)):
        pos_wall = []
        neg_wall = []
        for p in positive_positions:
            pos_wall.append(is_wall(p + i))
        final_check_pos = all(pos_wall) or not any(pos_wall) # make sure they are either all TRUE or all FALSE
        for p in negative_positions:
            neg_wall.append(is_wall(p + i))
        final_check_neg = all(neg_wall) or not any(neg_wall) # make sure they are either all TRUE or all FALSE
        if final_check_neg and final_check_pos:
            result = i
            break
    return result


def compute_props_list(action_samples, action: int):
    return ['WALL']


def algorithm_3():
    _right = 101
    _left = 100
    #ground_truth = gen_ground_truth()
    # states under which the action was NOT taken
    left_action_negative_samples = [{'StateRobotAct': _left,  'StateRobotPos': 0},
                                    {'StateRobotAct': _right,
                                     'StateRobotPos': -100},
                                    {'StateRobotAct': _left,
                                     'StateRobotPos': 485},
                                    {'StateRobotAct': _left,
                                     'StateRobotPos': -495}
                                    ]
    right_action_negative_samples = [{'StateRobotAct': _right, 'StateRobotPos': 495}
                                     ]
    left_action_positive_samples = [{'StateRobotAct': _right, 'StateRobotPos': 491}
                                    ]
    right_action_positive_samples = [{'StateRobotAct': _left,  'StateRobotPos': 0},
                                     {'StateRobotAct': _right, 'StateRobotPos': 1},
                                     {'StateRobotAct': _right,
                                         'StateRobotPos': 100},
                                     {'StateRobotAct': _right,
                                         'StateRobotPos': 485},
                                     {'StateRobotAct': _right,
                                         'StateRobotPos': -499}
                                     ]

    action_samples = {_left:  {'+': left_action_positive_samples,
                               '-': left_action_negative_samples},
                      _right: {'+': right_action_positive_samples,
                               '-': right_action_negative_samples}}

    max_vision = compute_max_vision(action_samples, _right)
    props_list = compute_props_list(action_samples, _right)

    i = 0
    for prog in ASP.__param_enumerate_1__(max_vision=max_vision, props_list=props_list):
        print('-'*75)
        i += 1
        consistent_with_demos = True
        print('asp_'+str(i) + ':\n   '+str(prog).replace('\n', '\n   '))
        # check samples
        for a in {_right, _left}:
            if not all(prog.eval(p) == a for p in action_samples[a]['+']) or not all(prog.eval(p) != a for p in action_samples[a]['-']):
                print('Discarded. Candidate program is not consistent with demos')
                consistent_with_demos = False
                break
        # this will save us a call to the ultimate because we already know the candidate ASP matches one of the negative demos
        if not consistent_with_demos:
            continue

        # make a call to Ultimate to check consistency with the safety spec
        b, trace = verifies(cstr(prog), get_counterexample=True)

        if b:  # the candidate program is consistent with the spec
            print('#'*75)
            print('Found a program consistent with the specification and demonstrations!')
            # an ASP consistent with the spec is found
            print(f"Satisfying Program:\n")
            print('asp_'+str(i) + ':\n   '+str(prog).replace('\n', '\n   '))
            break
        # the candidate program violates the spec (get a counter-example and add it to the negative samples)
        else:
            #demo = tuple(grouped2(trace))
            assert len(
                trace) == 3, 'for now we are assuming a single state transition in all demos'
            # parse the demo into a negative sample
            op = trace[1]
            s0 = trace[0]
            action_samples[op]['-'].append(s0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=int)
    parser.add_argument('--file', type=str,
                        default='descriptions/1d-hallway.yml')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    config = open_config_file(args.file)
    get_terminals_from_config(config)
    with open('verifier/model_prog.c', 'w') as f:
        f.write(str(make_model_program(config)))

    if args.alg == 1:
        algorithm_1()
    elif args.alg == 2:
        algorithm_2()
    else:
        print('running algorithm 3')
        algorithm_3()
