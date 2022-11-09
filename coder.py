#!/usr/bin/python3

from csnake import (
    CodeWriter, Function, Variable
)

class State(object):
    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

inverse_var_map = dict()
var_map = dict()

def define_imports_and_extern_lines():
    ''' Defines basic imports and externs required for programs to run '''
    cw = CodeWriter()
    cw.add_line('#include <stdio.h>')
    cw.add_lines([
        'extern void __VERIFIER_error() __attribute__ ((__noreturn__));',
        'extern void __VERIFIER_assume() __attribute__ ((__noreturn__));',
        'extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));',
    ])
    return cw

def define_static_property_function_lines(config):
    ''' Defines all the check_prop_X(Position) functions for every static property X '''
    cw = CodeWriter()
    for prop in config['StaticProperty']:
        bit = 1 if prop['include'] else 0

        func = Function(f"check_prop_{prop['name']}", 'int')
        for box in prop['ranges'][0]:
            for k in box:
                func.add_argument((k, 'int'))

        for box in prop['ranges']:
            cond = []
            for k, rnge in box.items():
                if rnge[0] == rnge[1]:
                    cond.append(f'({k} == {rnge[0]})')
                else:
                    cond.append(f'({k} >= {rnge[0]} && {k} <= {rnge[1]})')
            
            func.add_code(f"if ({' && '.join(cond)}) return {bit};")
        
        func.add_code(f'return {1-bit};')
        cw.add_function_definition(func)
    return cw

def define_vector_add(config):
    # <TODO> make it work with dimension
    func = Function('vector_add', 'int')
    func.add_arguments([('x', 'int'), ('d', 'int')])
    func.add_code('return x + d;')
    cw = CodeWriter()
    cw.add_function_definition(func)
    return cw

def define_action_enums_lines(config):
    cw = CodeWriter()
    for act in config['Action']:
        v = Variable(act['name'], 'int', value=act['value'])
        inverse_var_map[int(act['value'])] = act['name']
        var_map[act['name']] = int(act['value'])
        cw.add_line(str(v.generate_initialization()))
    return cw

def define_action_semantics_lines(config):
    func = Function('update_pos_from_action', 'int')
    func.add_argument(('act', 'int'))
    # <TODO> add arguments for supporting 
    func.add_argument(('x', 'int'))
    for act in config['Action']:
        # <TODO> parse act['update'] from dsl form to C form
        func.add_code(f"if (act == {act['name']}) return {act['update']};")
    cw = CodeWriter()
    cw.add_function_definition(func)
    return cw

def define_state_variables_lines(config):
    cw = CodeWriter()
    for var in config['State']:
        v = Variable(var['name'], var['type'])
        cw.add_variable_declaration(v)
    return cw

def define_initialization_lines(config):
    func = Function('initialize', 'void')
    for var in config['State']:
        if var['init'] == 'random' and 'ranges' in var:
            func.add_code(f'{var["name"]} = __VERIFIER_nondet_int();')
            cond = []
            for box in var['ranges']:
                for _, rnge in box.items():
                    if rnge[0] == rnge[1]:
                        cond.append(f'({var["name"]} == {rnge[0]})')
                    else:
                        cond.append(f'({var["name"]} >= {rnge[0]} && {var["name"]} <= {rnge[1]})')
            func.add_code(f'__VERIFIER_assume(({" || ".join(cond)}));')
        elif var['init'] == 'random' and 'values' in var:
            func.add_code(f'{var["name"]} = __VERIFIER_nondet_int();')
            cond = []
            for val in var['values']:
                cond.append(f'({var["name"]} == {val})')
            func.add_code(f'__VERIFIER_assume(({" || ".join(cond)}));')
        elif var['init'] in var_map:
            v = Variable(var['name'], var['type'], value=Variable(var['init'], var['type']))
            func.add_code(v.generate_initialization())
        else:
            v = Variable(var['name'], var['type'], value=var['init'])
            func.add_code(v.generate_initialization())
    cw = CodeWriter()
    cw.add_function_definition(func)
    return cw

def define_policy_lines():
    cw = CodeWriter()
    func_policy = Function('policy', 'int')
    func_policy.add_code('INSERT_ASP')
    func_policy.add_code('return RIGHT;')
    cw.add_function_definition(func_policy)
    return cw

def define_spec_compute_lines(config):
    cw = CodeWriter()
    func = Function('compute_spec', 'void')
    for spec in config['Specs']:
        v = Variable(spec['name'], 'int')
        cw.add_variable_declaration(v)
        # <TODO> parse spec statement from a DSL statement
        func.add_code(f'{spec["name"]} = {spec["value"]};')
    cw.add_function_definition(func)
    return cw

def define_main_lines(config):
    func = Function('main', 'int')
    func.add_code('initialize();')
    func.add_code('compute_spec();')
    for spec in config['Specs']:
        func.add_code(f'//@ assert ({spec["name"]} != 0);')
    func.add_code('while (1) {')
    func.add_code('    StateRobotAct = policy();')
    func.add_code('    StateRobotPos = update_pos_from_action(StateRobotAct, StateRobotPos);')
    func.add_code('    compute_spec();')
    for spec in config['Specs']:
        func.add_code(f'    //@ assert ({spec["name"]} != 0);')
    func.add_code('}')
    cw = CodeWriter()
    cw.add_function_definition(func)
    return cw

def make_model_program(config):

    cw = CodeWriter()

    cw.add_lines(define_imports_and_extern_lines())
    cw.add_line()
    cw.add_lines(define_vector_add(config))
    cw.add_line()
    cw.add_lines(define_action_enums_lines(config))
    cw.add_line()
    cw.add_lines(define_action_semantics_lines(config))
    cw.add_line()
    cw.add_lines(define_static_property_function_lines(config))
    cw.add_line()
    cw.add_lines(define_state_variables_lines(config))
    cw.add_line()
    cw.add_lines(define_initialization_lines(config))
    cw.add_line()
    cw.add_lines(define_policy_lines())
    cw.add_line()
    cw.add_lines(define_spec_compute_lines(config))
    cw.add_line()
    cw.add_lines(define_main_lines(config))
    cw.add_line()

    return cw

if __name__ == '__main__':
    from utils import open_config_file
    config = open_config_file('descriptions/1d-hallway.yml')
    cw = make_model_program(config)
    print(cw)