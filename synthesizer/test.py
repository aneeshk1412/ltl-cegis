#!/usr/bin/python3
__author__ = "Kia Rahmani"


"""A simple test program which repeatedly creates and prints programs using a synthesizer """
import sys
from src.synthesizer import *

def main(arguments):
    synth = Synthesizer()
    asps = synth.enumerate_all_asps()
    print (len(asps), ' ASPs of length=1 are generated')
    # pick an ASP and print it
    print(asps[10].pretty_str())



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
