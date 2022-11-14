#!/usr/bin/python3

import os
import subprocess

def grouped2(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def abs_path(filename):
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_directory, filename)

def parse_by_filter(filter_func, line_split):
    d = dict(x.split('=') for x in filter(filter_func, line_split))
    return {k: int(v) for k, v in d.items()}

def parse_to_trace(ultimate_trace):
    bit = ''
    for line in ultimate_trace:
        if line.find('RET') != -1 and line.find('compute_atomic_propositions()') != -1:
            bit = 'compute_atomic_propositions'
        elif bit == 'compute_atomic_propositions':
            line_split = line.strip().translate(str.maketrans('', '', '][,')).strip().split()

            action_filter = lambda x: x.find('\\') == -1 and x.find('StateRobotAct') != -1
            yield parse_by_filter(action_filter, line_split)

            state_filter = lambda x: x.find('\\') == -1 and x.find('State') != -1
            yield parse_by_filter(state_filter, line_split)
            bit = ''

def parse_result(ultimate_result):
    if ultimate_result.find('RESULT: Ultimate proved your program to be correct!') != -1:
        return True, None
    if ultimate_result.find('RESULT: Ultimate proved your program to be incorrect!') != -1:
        ultimate_trace = ultimate_result[ultimate_result.find('Results'):].splitlines()
        return False, list(parse_to_trace(ultimate_trace))
    raise NotImplementedError

def call_ultimate(prog_abs_path, property='safety'):
    if property == 'safety':
        tc_file = abs_path('config/AutomizerReach.xml')
        s_file = abs_path('config/svcomp-Reach-64bit-Automizer_Default.epf')
    elif property == 'liveness':
        tc_file = abs_path('config/AutomizerLTL.xml')
        s_file = abs_path('config/svcomp-LTL-64bit-Automizer_Default.epf')

    result = subprocess.run(['Ultimate', '-tc', tc_file, '-s', s_file, '-i', prog_abs_path], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    return result

def verifies(c_model_prog, c_asp, property='safety'):
    c_prog = c_model_prog.replace('INSERT_ASP', c_asp)
    with open(abs_path('tempprog.c'), 'w') as fp:
        fp.write(c_prog)
    result = call_ultimate(abs_path('tempprog.c'), property)
    return parse_result(result)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, help='Filename')
    parser.add_argument('--p', type=str, help='Property')
    args = parser.parse_args()
    result = call_ultimate(abs_path(args.f), args.p)
    print(result)
    b, trace = parse_result(result)
    print(b)
    d = {102: 'LEFT', 101: 'RIGHT', 100: 'NONE'}
    if trace:
        for item in grouped2(trace):
            act, state = item
            act['StateRobotAct'] = d[act['StateRobotAct']]
            state['StateRobotAct'] = d[state['StateRobotAct']]
            print(act, state)
