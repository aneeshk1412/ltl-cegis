import json
from csnake import (
    CodeWriter, Enum, Function, Variable
)

def parse_imports_and_extern():
    cw = CodeWriter()
    cw.add_line("#include <stdio.h>")
    cw.add_line("")
    cw.add_lines([
        "extern void __VERIFIER_error() __attribute__ ((__noreturn__));",
        "extern void __VERIFIER_assume() __attribute__ ((__noreturn__));",
        "extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));",
    ])
    cw.add_line("")
    return cw


def parse_position_properties(config):
    cw = CodeWriter()
    for p in config["position properties"]:
        b = 1 if p['include'] else 0

        func = Function(f"check_{p['name']}", "int")
        for dim in range(config['dimensions']):
            func.add_argument((f"x_{dim}", "int"))
        for rnge in p['ranges']:
            cond = []
            for dim in range(config['dimensions']):
                if rnge[dim][0] == rnge[dim][1]:
                    cond.append(f"(x_{dim} == {rnge[dim][0]})")
                else:
                    cond.append(f"(x_{dim} >= {rnge[dim][0]} && x_{dim} <= {rnge[dim][1]})")
            func.add_code(f"if ({' && '.join(cond)}) return {b};")
        func.add_code(f"return {1-b};")
        cw.add_function_definition(func)
        cw.add_line("")
    return cw

vars_map = dict()

def parse_initialization(config):
    cw = CodeWriter()
    main_init_list = []

    for key in config.keys():
        varlist = []
        if key == "robot" or key.startswith("agent"):
            for ele in config[key]["state"]:
                if ele['type'] in ['action']:
                    ele['type'] = 'int'
                var = Variable(f"{key}_state_{ele['name']}", ele['type'])
                varlist.append(var)
                vars_map[var.name] = ele['dsl_name']

                if ele['init'] == "random":
                    func = Function(f"randint_{key}_state_{ele['name']}", "int")
                    for rnge in ele['rand_ranges']:
                        cond = []
                        for dim in range(config['dimensions']):
                            if rnge[dim][0] == rnge[dim][1]:
                                cond.append(f"(x == {rnge[dim][0]})")
                            else:
                                cond.append(f"(x >= {rnge[dim][0]} && x <= {rnge[dim][1]})")
                    func.add_code([
                        "int x = __VERIFIER_nondet_int();",
                        f"__VERIFIER_assume(({' || '.join(cond)}));",
                        "return x;"
                    ])
                    cw.add_function_definition(func)
                    cw.add_line("")
                    main_init_list.append(f"{var.name} = {func.generate_call()};")
                else:
                    main_init_list.append(f"{var.name} = {ele['init']};")

            for var in varlist:
                cw.add_line(var.generate_declaration() + ";")
            cw.add_line("")

    func_init = Function('initialize', 'void')
    func_init.add_code(main_init_list)
    cw.add_function_definition(func_init)
    cw.add_line("")
    return cw


def parse_enum(config):
    cw = CodeWriter()
    ctr = 100
    for t in config["type interpretations"]:
        if t["type"] == "enum":
            # enum = Enum(f"{t['name']}", prefix=None, typedef=True)
            # for val in t["range"]:
            #     enum.add_value(val["name"])
            # cw.add_enum(enum)
            # cw.add_line("")
            for val in t["range"]:
                cw.add_line(f"int {val['name']} = {ctr};")
                vars_map[ctr] = val['name']
                ctr += 1
            cw.add_line("")
    return cw


def parse_policy(policy):
    return policy


def get_c_program(description_file):
    with open(description_file, "rb") as f:
        lines = f.read()
        config = json.loads(lines)

    cw = CodeWriter()
    cw.add_lines(parse_imports_and_extern())
    cw.add_lines(parse_position_properties(config))
    cw.add_lines(parse_enum(config))
    cw.add_lines(parse_initialization(config))

    func_policy = Function('policy', 'void')
    func_policy.add_code(parse_policy(config['robot']['policy']))
    cw.add_function_definition(func_policy)

    func_update = Function('update', 'void')
    func_update.add_code(["if (robot_state_act == LEFT) robot_state_pos = robot_state_pos - 1;",
        "if (robot_state_act == RIGHT) robot_state_pos = robot_state_pos + 1;",
        "if (robot_state_act == NONE) robot_state_pos = robot_state_pos;"])
    cw.add_function_definition(func_update)

    func_precompute = Function('precompute', 'void')
    varlist = []
    for comps in config['precompute']:
        vars_map[comps['name']] = comps['exp']
        var = Variable(comps['name'], comps['type'])
        varlist.append(var.generate_declaration() + ";")
        func_precompute.add_code([f"{comps['name']} = {comps['exp']};"])
    cw.add_lines(varlist)
    cw.add_function_definition(func_precompute)

    func_main = Function('main', 'int')
    func_main.add_code([
        "initialize();",
        "precompute();",
        "while (1) {",
        "    policy();",
        "    update();",
        "    precompute();",
    ])
    for spec in config['specification']:
        func_main.add_code(f"    {spec}")
    func_main.add_code("}")
    cw.add_function_definition(func_main)

    return cw, vars_map

if __name__ == "__main__":
    get_c_program("../descriptions/structure.json")
