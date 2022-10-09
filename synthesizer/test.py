#!/usr/bin/python3
__author__ = "Kia Rahmani"


"""A simple test program which repeatedly creates and prints programs using a synthesizer """
import sys
import random
from src.synthesizer import *

def main(arguments):
    synth = Synthesizer()
    asps = synth.enumerate_all_asps()
    number_of_asps = len(asps)
    print (number_of_asps, ' ASPs of length=1 are generated')
    # print 1000 randomly chosen programs 
    for i in range(0,1000):
        index = random.randint(0, number_of_asps - 1)
        print(asps[index].pretty_str())
    



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


class Spec:
    pass