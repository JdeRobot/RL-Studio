
from datetime import datetime
import time
import sys
import argparse
from cprint import cprint
from functools import reduce
import yaml

import gym
import numpy as np
import os
from icecream import ic

# Importing app local files
from rl_studio.agents.f1 import settings
from rl_studio.agents.f1 import utils
from rl_studio.agents.f1 import liveplot
from rl_studio.agents.f1.brains.train_qlearn import train_qlearn


ic.enable()
ic.configureOutput(prefix=f'{datetime.now()} | ')

def read_config(yaml_file):
    '''
    read and parse YAML file with config
    '''
    with open(yaml_file, 'r') as stream:
        config_yaml = yaml.safe_load(stream)
    
    return config_yaml

def main():
    # ------------------- Parameter parsing from YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()
    config = read_config(args.config_file)
    execute_algor = f"{config['Method']}_{config['Algorithm']}"

    ic(execute_algor)

    # ------------------- CREATE DIRS
    os.makedirs("logs", exist_ok=True)
    os.makedirs("images", exist_ok=True)
           
    #if execute_algor == 'train_dql':
    #    train_dql(config)   

    if execute_algor == 'train_qlearn':
        train_qlearn()           


if __name__ == '__main__':

    main()
