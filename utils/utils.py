#!/usr/bin/python3

import yaml

def open_config_file(configfile):
    with open(configfile, 'r') as stream:
        return yaml.safe_load(stream)
