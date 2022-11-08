#!/usr/bin/python3

import yaml
from csnake import (
    CodeWriter, Function, Variable, Struct
)

def open_config_file(configfile):
    with open(configfile, 'r') as stream:
        return yaml.safe_load(stream)


def define_imports_and_extern():
    ''' Defines basic imports and externs required for programs to run '''
    cw = CodeWriter()
    cw.add_line('#include <stdio.h>')
    cw.add_line('')
    cw.add_lines([
        'extern void __VERIFIER_error() __attribute__ ((__noreturn__));',
        'extern void __VERIFIER_assume() __attribute__ ((__noreturn__));',
        'extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));',
    ])
    return cw


def define_vector_struct(config):
    vec = Struct('Vector', typedef=True)
    for e in config['Dimensions']:
        vec.add_variable((e, 'int'))
    return vec.generate_declaration()


def define_vector_add(config):
    func = Function('vector_add', 'Vector')
    func.add_arguments([('p', 'Vector')] + [(f'd{e}', 'int') for e in config['Dimensions']])
    for e in config['Dimensions']:
        func.add_code(f'p.{e} = p.{e} + d{e};')
    func.add_code('return p;')
    return func.generate_definition()


def define_static_property_function(config):
    ''' Defines all the check_prop_X(Position) functions for every static property X '''
    cw = CodeWriter()
    for prop in config['StaticProperty']:
        bit = 1 if prop['include'] else 0

        func = Function(f"check_prop_{prop['name']}", 'int')
        func.add_argument(('p', 'Vector'))

        for box in prop['ranges']:
            cond = []
            for k, rnge in box.items():
                match rnge:
                    case int():
                        cond.append(f'(p.{k} == {rnge})')
                    case [int(), int()]:
                        cond.append(f'(p.{k} >= {rnge[0]} && p.{k} <= {rnge[1]})')
            func.add_code(f"if ({' && '.join(cond)}) return {bit};")
        
        func.add_code(f'return {1-bit};')
        cw.add_function_definition(func)
    return cw


def define_action_enums(config):
    cw = CodeWriter()
    for act in config['Action']:
        v = Variable(act['name'], 'int', value=act['value'])
        cw.add_line(str(v.generate_initialization()))
    return cw


def define_action_semantics(config):
    func = Function('update_pos_from_action', 'Vector')
    func.add_arguments([('act', 'int'), ('p', 'Vector')])
    for act in config['Action']:
        # <TODO> parse act['update'] from dsl form to C form
        func.add_code(f"if (act == {act['name']}) {act['update']};")
    func.add_code('return p;')
    return func.generate_definition()


def define_state_variables(config):
    cw = CodeWriter()
    for var in config['State']:
        v = Variable(var['name'], var['type'])
        cw.add_variable_declaration(v)
    return cw


def define_initialization(config):
    func = Function('initialize', 'void')
    for var in config['State']:
        if var['type'] == 'Vector':
            for k in config['Dimensions']:
                func.add_code(f'{var["name"]}.{k} = __VERIFIER_nondet_int();')
            cond = []
            for box in var['ranges']:
                cond_box = []
                for k, rnge in box.items():
                    match rnge:
                        case int():
                            cond_box.append(f'({var["name"]}.{k} == {rnge})')
                        case [int(), int()]:
                            cond_box.append(f'({var["name"]}.{k} >= {rnge[0]} && {var["name"]}.{k} <= {rnge[1]})')
                cond.append('(' + " && ".join(cond_box) + ')')
            func.add_code(f'__VERIFIER_assume(({" || ".join(cond)}));')
        if var['type'] == 'int':
            func.add_code(f'{var["name"]} = __VERIFIER_nondet_int();')
            cond = []
            for rnge in var['ranges']:
                match rnge:
                    case int() | str():
                        cond.append(f'({var["name"]} == {rnge})')
                    case [int(), int()]:
                        cond.append(f'({var["name"]} >= {rnge[0]} && {var["name"]}.{k} <= {rnge[1]})')
            func.add_code(f'__VERIFIER_assume(({" || ".join(cond)}));')
    cw = CodeWriter()
    cw.add_function_definition(func)
    return cw


def define_policy():
    cw = CodeWriter()
    func_policy = Function('policy', 'int')
    func_policy.add_code('INSERT_ASP')
    func_policy.add_code('return NONE;')
    cw.add_function_definition(func_policy)
    return cw


def define_asserts(config, indents=0):
    cw = CodeWriter()
    for _ in range(indents):
        cw.indent()
    for aspec in config['AssertSpecs']:
        cw.add_line(f'if ({aspec}) {{')
        cw.indent()
        cw.add_line('__VERIFIER_error();')
        cw.dedent()
        cw.add_line('}')
    for _ in range(indents):
        cw.dedent()
    return cw


def define_compute_atomic_propositions(config):
    cw = CodeWriter()
    func = Function('compute_atomic_propositions', 'void')
    for spec in config['AtomicPropositions']:
        v = Variable(spec['name'], 'int', value=spec['init'])
        cw.add_variable_initialization(v)
        # <TODO> parse spec statement from a DSL statement
        func.add_code(f'{spec["name"]} = {spec["value"]};')
    cw.add_function_definition(func)
    return cw


def define_spec(config):
    cw = CodeWriter()
    cw.add_line(f'//@ ltl invariant positive: {config["LTLSpec"]};')
    return cw


def define_main(config):
    func = Function('main', 'int')
    func.add_code('initialize();')
    func.add_code('compute_atomic_propositions();')
    func.add_code(define_asserts(config, 0))
    func.add_code('while (1) {')
    func.add_code('    StateRobotAct = policy();')
    func.add_code('    StateRobotPos = update_pos_from_action(StateRobotAct, StateRobotPos);')
    func.add_code('    compute_atomic_propositions();')
    func.add_code(define_asserts(config, 1))
    func.add_code('}')
    cw = CodeWriter()
    cw.add_function_definition(func)
    return cw


def make_model_program(config):

    cw = CodeWriter()

    cw.add_lines(define_spec(config))
    cw.add_line()
    cw.add_lines(define_imports_and_extern())
    cw.add_line()
    cw.add_lines(define_vector_struct(config))
    cw.add_line()
    cw.add_lines(define_vector_add(config))
    cw.add_line()
    cw.add_lines(define_action_enums(config))
    cw.add_line()
    cw.add_lines(define_action_semantics(config))
    cw.add_line()
    cw.add_lines(define_static_property_function(config))
    cw.add_line()
    cw.add_lines(define_state_variables(config))
    cw.add_line()
    cw.add_lines(define_initialization(config))
    cw.add_line()
    cw.add_lines(define_policy())
    cw.add_line()
    cw.add_lines(define_compute_atomic_propositions(config))
    cw.add_line()
    cw.add_lines(define_main(config))
    cw.add_line()

    return cw

if __name__ == '__main__':
    config = open_config_file('../descriptions/1d-patrolling.yml')
    cw = make_model_program(config)
    print(cw)