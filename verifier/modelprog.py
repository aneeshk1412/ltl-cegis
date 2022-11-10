#!/usr/bin/python3

from itertools import count
from csnake import (
    CodeWriter, Function, Variable
)

global_counter = count(100)

def define_liveness_spec(config):
    ''' <TODO> Parse the liveness spec from DSL kind language '''
    cw = CodeWriter()
    cw.add_line(f'//@ ltl invariant positive: {config["livenessSpec"]};')
    return cw

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

def variables_action_enums(config):
    d = dict()
    for _, actions_dict in config['actionTypes'].items():
        d.update({action: Variable(action, 'int', value=c) for action, c in zip(actions_dict['actions'], global_counter)})
    return d

def variables_state(config):
    mapping = dict()
    for statevar, details in config['state'].items():
        if details['type'] == 'Vector':
            vector_map = dict()
            for dim in config['dimensions']:
                vector_map[dim] = Variable(statevar + dim, 'int')
            mapping[statevar] = vector_map
        elif details['type'] in config['actionTypes']:
            mapping[statevar] = Variable(statevar, 'int')
        else:
            mapping[statevar] = Variable(statevar, details['type'])
    return mapping

def function_action_semantics_for_act_var(config, act_var):
    var_mapping = config['state'][act_var]['mapping']
    var_type = config['state'][act_var]['type']
    func = Function(f'apply_{act_var}', 'void')
    lines = ''
    for action, details in config['actionTypes'][var_type]['actions'].items():
        # <TODO> parse details['update'] from dsl form to C form
        lines += f"if ({act_var} == {action}) {details['update']};\n"
    for k, v in var_mapping.items():
        lines = lines.replace(k + '.', v)
    func.add_code(lines)
    print(func.generate_definition())
    return func

def function_static_property_function(prop, details, config):
    ''' Returns check_prop_X(Position) function for static property X '''
    bit = 1 if details['include'] else 0
    func = Function(f"check_prop_{prop}", 'int', arguments=[(f'p{dim}', 'int') for dim in config['dimensions']])
    for box in details['ranges']:
        cond = []
        for k, rnge in box.items():
            match rnge:
                case int():
                    cond.append(f'(p{k} == {rnge})')
                case [int(), int()]:
                    cond.append(f'(p{k} >= {rnge[0]} && p{k} <= {rnge[1]})')
        func.add_code(f'if ({" && ".join(cond)}) return {bit};')
    func.add_code(f'return {1-bit};')
    return func

def define_atomic_propositions(config):
    cw = CodeWriter()
    for propos, details in config['atomicPropositions'].items():
        v = Variable(propos, 'int', value=details['init'])
        cw.add_variable_initialization(v)
    return cw

def function_compute_atomic_propositions(config):
    func = Function('compute_atomic_propositions', 'void')
    for propos, details in config['atomicPropositions'].items():
        # <TODO> parse spec statement from a DSL statement
        func.add_code(f'{propos} = {details["value"]};')
    return func

def function_policy():
    func_policy = Function('policy', 'int')
    func_policy.add_code('INSERT_ASP')
    func_policy.add_code('return NONE;')
    return func_policy

def function_initialization(config):
    func = Function('initialize', 'void')
    for statevar, details in config['state'].items():
        if details['type'] == 'vector':
            for dim in config['dimensions']:
                func.add_code(f'{statevar}{dim} = __VERIFIER_nondet_int();')
            cond = []
            for box in details['ranges']:
                cond_box = []
                for dim, rnge in box.items():
                    match rnge:
                        case int():
                            cond_box.append(f'({statevar}{dim} == {rnge})')
                        case [int(), int()]:
                            cond_box.append(f'({statevar}{dim} >= {rnge[0]} && {statevar}{dim} <= {rnge[1]})')
                cond.append('(' + " && ".join(cond_box) + ')')
            func.add_code(f'__VERIFIER_assume(({" || ".join(cond)}));')
        elif details['type'] in config['actionTypes']:
            func.add_code(f'{statevar} = __VERIFIER_nondet_int();')
            cond = []
            for act in config['actionTypes'][details['type']]['actions']:
                cond.append(f'{statevar} == {act}')
            func.add_code(f'__VERIFIER_assume(({" || ".join(cond)}));')
        elif details['type'] == 'int':
            func.add_code(f'{statevar} = __VERIFIER_nondet_int();')
            cond = []
            for rnge in details['ranges']:
                match rnge:
                    case int() | str():
                        cond.append(f'({statevar} == {rnge})')
                    case [int(), int()]:
                        cond.append(f'({statevar} >= {rnge[0]} && {details["name"]}.{dim} <= {rnge[1]})')
            func.add_code(f'__VERIFIER_assume(({" || ".join(cond)}));')
    return func

def add_safety_specs_to_cw(cw, config):
    for aspec in config['safetySpecs']:
        cw.add_line(f'//@assert ({aspec});')

def make_model_program(config):
    cw = CodeWriter()

    cw.add_lines(define_liveness_spec(config))
    cw.add_line()
    cw.add_lines(define_imports_and_extern())
    cw.add_line()

    action_vars_map = variables_action_enums(config)
    for var in action_vars_map.values():
        cw.add_variable_initialization(var)
    cw.add_line()

    state_vars_map = variables_state(config)
    for x in state_vars_map.values():
        if isinstance(x, dict):
            for y in x.values():
                cw.add_variable_declaration(y)
        else:
            cw.add_variable_declaration(x)
    cw.add_line()

    actions = ['StateRobotAct']
    action_semantic_funcs = [function_action_semantics_for_act_var(config, v) for v in actions]
    for func in action_semantic_funcs:
        cw.add_function_definition(func)
        cw.add_line()

    static_prop_funcs = [function_static_property_function(stprop, details, config) for stprop, details in config['staticProperty'].items()]
    for func in static_prop_funcs:
        cw.add_function_definition(func)
    cw.add_line()

    cw.add_lines(define_atomic_propositions(config))
    cw.add_line()
    compute_atprop_func = function_compute_atomic_propositions(config)
    cw.add_function_definition(compute_atprop_func)
    cw.add_line()

    policy_func = function_policy()
    cw.add_function_definition(policy_func)
    cw.add_line()

    init_func = function_initialization(config)
    cw.add_function_definition(init_func)
    cw.add_line()

    main_cw = CodeWriter()
    main_cw.add_function_call(init_func)
    main_cw.add_function_call(compute_atprop_func)
    add_safety_specs_to_cw(main_cw, config)

    main_cw.add_line('while (1)')
    main_cw.open_brace()
    main_cw.add_line(f'StateRobotAct = {policy_func.generate_call()};')
    for func in action_semantic_funcs:
        main_cw.add_line(f'{func.generate_call()};')
    main_cw.add_function_call(compute_atprop_func)
    add_safety_specs_to_cw(main_cw, config)
    main_cw.close_brace()

    main_func = Function('main', 'int')
    main_func.add_code(main_cw)
    cw.add_function_definition(main_func)
    cw.add_line()

    return cw

if __name__ == '__main__':
    import yaml
    with open('../descriptions/2d-patrolling.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    cw = make_model_program(config)
    print(cw)
