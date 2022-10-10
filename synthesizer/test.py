#!/usr/bin/python3
__author__ = "Kia Rahmani"


"""A simple test program which repeatedly creates and prints programs using a synthesizer """
from os import defpath
import sys
import random
from src.synthesizer import *


def main(arguments):
    synth = Synthesizer()
    policy_depth = 2
    asps = synth.enumerate_all_asps(depth=policy_depth, cap=4000)
    number_of_asps = len(asps)
    for asp in asps:
        print(asp.pretty_str())
        print (50*'-')
    print (number_of_asps, ' ASPs of length =', policy_depth,'are generated')

        
    


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


class Spec:
    pass