import json
from csnake import (
    CodeWriter, Enum, Function, Variable
)

with open("descriptions/structure.json", "rb") as f:
    lines = f.read()
    config = json.loads(lines)

# print(config)

cw = CodeWriter()

cw.add_line("#include <stdio.h>")
cw.add_line("")
cw.add_lines([
    "extern void __VERIFIER_error() __attribute__ ((__noreturn__));", 
    "extern void __VERIFIER_assume() __attribute__ ((__noreturn__));",
    "extern int __VERIFIER_nondet_int() __attribute__ ((__noreturn__));",
])
cw.add_line("")

''' Parse position properties '''
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

for t in config["type interpretations"]:
    if t["type"] == "enum":
        enum = Enum(f"{t['name']}", prefix=None, typedef=True)
        for val in t["range"]:
            enum.add_value(val["name"])
        cw.add_enum(enum)
        cw.add_line("")

main_init_list = []

for key in config.keys():
    varlist = []
    if key == "robot" or key.startswith("person"):
        for ele in config[key]["state"]:
            var = Variable(f"{key}_state_{ele['name']}", ele['type'])
            varlist.append(var)

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
                    f"__VERIFIER_assume(({' || '.join(cond)}));"
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

def parse_policy(policy):
    return policy

func_policy = Function('policy', 'void')
func_policy.add_code(parse_policy(config['robot']['policy']))
cw.add_function_definition(func_policy)
cw.add_line("")

func_update = Function('update', 'void')
func_update.add_code(["if (robot_state_act == LEFT) robot_state_pos = robot_state_pos - 1;",
    "if (robot_state_act == RIGHT) robot_state_pos = robot_state_pos + 1;",
    "if (robot_state_act == NONE) robot_state_pos = robot_state_pos;"])
cw.add_function_definition(func_update)
cw.add_line("")

func_main = Function('main', 'int')
func_main.add_code([
    "initialize();",
    "while (1) {",
    "   policy();",
    "   update();",
    "}"
])
cw.add_function_definition(func_main)
cw.add_line("")

print(cw)
