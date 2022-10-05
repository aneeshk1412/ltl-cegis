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
                cond.append(f"x_{dim} == {rnge[dim][0]}")
            else:
                cond.append(f"x_{dim} >= {rnge[dim][0]} && x_{dim} <= {rnge[dim][1]}")
        func.add_code(f"if ({' && '.join(cond)}) return {b};")
    func.add_code(f"return {1-b};")
    cw.add_function_definition(func)
    cw.add_line("")

for t in config["type interpretations"]:
    if t["type"] == "enum":
        enum = Enum(f"{t['name']}", prefix=t['name']+"_", typedef=True)
        for val in t["range"]:
            enum.add_value(val["name"])
        cw.add_enum(enum)
        cw.add_line("")

print(cw)
