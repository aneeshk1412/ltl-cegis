#!/usr/bin/python3

from inspect import trace
import os
import subprocess

package_directory = os.path.dirname(os.path.abspath(__file__))
def abs_path(filename):
    return os.path.join(package_directory, filename)

def parse_action(action_line):
    action_line = action_line.strip().translate(str.maketrans('', '', '][,')).strip().split()
    filter_func = lambda x: x.find('\\') == -1 and x.find('Act') != -1 and x.find('State') != -1
    return dict(x.split('=') for x in filter(filter_func, action_line))

def parse_state(state_line):
    state_line = state_line.strip().translate(str.maketrans('', '', '][,')).strip().split()
    filter_func = lambda x: x.find('\\') == -1 and x.find('State') != -1
    return dict(x.split('=') for x in filter(filter_func, state_line))

def parse_to_c_trace(ultimate_trace):
    bit = ''
    for line in ultimate_trace:
        if line.find('StateRobotAct = policy()') != -1:
            bit = 'policy'
        elif bit == 'policy':
            yield parse_action(line)
            bit = ''
        elif line.find('RET') != -1 and line.find('compute_spec()') != -1:
            bit = 'compute_spec'
        elif bit == 'compute_spec':
            yield parse_state(line)
            bit = ''

def get_proven_or_trace(c_prog):
    with open(abs_path('tempprog.c'), 'w') as fp:
        fp.writelines(line + '\n' for line in c_prog)

    result = subprocess.run(['Ultimate', abs_path('LTLAutomizerC.xml'), abs_path('tempprog.c'), '--settings', abs_path('Default.epf')], stdout=subprocess.PIPE)

    os.remove(abs_path('tempprog.c'))
    result = result.stdout.decode('utf-8')
    # print(result)

    if result.find('assertion can be violated') == -1:
        return True, None

    trace = result[result.find('FailurePath:'):result.find('\n\n')].strip().splitlines()
    return False, list(parse_to_c_trace(trace))

def verifies(c_asp, get_counterexample=False):
    ## <TODO> Hardcoded spec and environment for now
    # print(dsl_asp)
    with open(abs_path('model_prog.c'), 'r') as fp:
        prog_string = fp.read()
    prog_string = prog_string.replace('INSERT_ASP', c_asp)
    if get_counterexample:
        b, trace = get_proven_or_trace(prog_string.splitlines())
        return b, trace
    else:
        b, _ = get_proven_or_trace(prog_string.splitlines())
        return b

if __name__ == '__main__':
    result = subprocess.run(['Ultimate', abs_path('LTLAutomizerC.xml'), abs_path('test_prog.c'), '--settings', abs_path('Default.epf')], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    print(result)

    if result.find('assertion can be violated') == -1:
        print(True, None)
    else:
        trace = result[result.find('FailurePath:'):result.find('\n\n')].strip().splitlines()
        print(False)
        for ele in list(parse_to_c_trace(trace)):
            print(ele)