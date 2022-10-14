#!/usr/bin/python3

from parse_description import get_c_program
import subprocess
import argparse

def get_type(s, t, vars_map):
    s = s.translate(str.maketrans('', '', '][,'))
    s = s.strip().split()
    res = []
    for ele in s:
        if ele.find(t) != -1 and ele.find(')') == -1:
            # l = ele.split('=')
            # cur = vars_map[l[0]] + " : "
            # if t == "act":
            #     cur += vars_map[l[1]]
            # else:
            #     cur += l[1]
            # res.append(cur)
            if t == "act":
                l = ele.split('=')
                l[1] = vars_map[int(l[1])]
                ele = '='.join(l)
            res.append(ele)
    return res

def get_trace(output, vars_map):
    if output.find("assertion can be violated") == -1:
        print("The program has been verified or no violation was found within the Time Limit")
    else:
        output = output.strip().split("\n")
        bit = ""
        for line in output:
            if (line.find("policy") != -1 and line.find("RET") != -1):
                bit = "policy"
                print("Action:")
            elif (line.find("precompute") != -1 and line.find("RET") != -1):
                bit = "precompute"
                print("State:")
            elif bit == "policy":
                print(get_type(line, "act", vars_map))
                bit = ""
            elif bit == "precompute":
                print(get_type(line, "pos", vars_map))
                bit = ""
            else:
                continue

def main(config_file):
    prog, vars_map = get_c_program(config_file)
    prog_lines = str(prog)
    with open("testprog.c", 'w') as file:
        for line in prog_lines:
            file.write(line)
    result = subprocess.run(['./Ultimate', 'LTLAutomizerC.xml', 'testprog.c', '--settings', 'Default.epf'], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    # print(result)
    get_trace(result, vars_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str)
    args = parser.parse_args()

    main(args.f)

