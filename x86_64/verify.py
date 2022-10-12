#!/usr/bin/python3

from parse_description import get_c_program
import subprocess
import argparse

def main(config_file):
    prog, vars_map = get_c_program(config_file)
    prog_lines = str(prog)
    with open("testprog.c", 'w') as file:
        for line in prog_lines:
            file.write(line)
    result = subprocess.run(['./Ultimate', 'LTLAutomizerC.xml', 'testprog.c', '--settings', 'Default.epf'], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    print(result)
    result = result.strip().split('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str)
    args = parser.parse_args()

    main(args.f)

