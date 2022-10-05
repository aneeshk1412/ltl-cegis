import json
from csnake import (
    CodeWriter, Enum, Function, Variable
)

with open("descriptions/structure.json", "rb") as f:
    lines = f.read()
    config = json.loads(lines)

# print(config)

cw = CodeWriter()

''' Parse position properties '''
for p in config["position properties"]:
    enum = Enum(f"IS_{p['name']}", prefix=p['name'], typedef=True)
    enum.add_value("_NO", 0)
    enum.add_value("_YES", 1)
    cw.add_enum(enum)
    cw.add_line("")

    b = 1 if p['type'] else 0

    func = Function(f"check_{p['name']}", "int")
    for dim in range(config['dimensions']):
        func.add_argument((f"x_{dim}", "int"))
    for rnge in p['ranges']:
        cond = ""
        for dim in range(config['dimensions']):
            cond += f"x_{dim} >= {rnge[dim][0]} && x_{dim} <= {rnge[dim][1]}"
        func.add_code(f"if ({cond}) return {b};")
    func.add_code(f"return {1-b};")
    cw.add_function_definition(func)
    cw.add_line("")

print(cw)
