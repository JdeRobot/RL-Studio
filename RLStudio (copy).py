import datetime
import time
import sys
import argparse
from cprint import cprint
from functools import reduce

import gym
import numpy as np

# next imports
# from agents.f1.settings
# from agents.f1.liveplot
# from agents.f1.utils
# from agents.f1.qlearn

# Importing app local files
import settings
import liveplot
import utils
import settings

from algorithms.qlearn import QLearn
from agents.f1.train_qlearning_f1 import train_qlearning_f1



if __name__ == '__main__':

    # Parameter parsing from YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()

    #config, algorithm1, algorithm_hyperparams1, model1, actions1, gaz_pos1 = utils.read_config(args.config_file)
    config = utils.read_config(args.config_file)

    #method = config['Method']
    #algorithm = config['Algorithm']
    #algorithm_hyperparams = config['Hyperparams']
    #model = config['Model']
    #agent = config['Agent']
    #actions = config['envs_params'][model]['actions']
    #set_actions = config[actions]
    #start_pose = [config['envs_params'][model]['start_pose_x'], config['envs_params'][model]['start_pose_y']]


    #print(f"\n [RLStudio.py] -> method: {method}")
    #print(f"\n [RLStudio.py] -> algorithm: {algorithm}")
    #print(f"\n [RLStudio.py] -> algorithm_hyperparams: {algorithm_hyperparams}")
    #print(f"\n [RLStudio.py] -> model: {model}")
    #print(f"\n [RLStudio.py] -> agent: {agent}")
    #print(f"\n [RLStudio.py] -> set_actions: {set_actions}")
    #print(f"\n [RLStudio.py] -> gaz_pos: {gaz_pos}")
    #print(f"\n [RLStudio.py] -> start_pose: {start_pose}")


    #print(f"\n config['Hyperparams in qlearning']: {config['qlearning']}")
    #print(f"\n config['Hyperparams in qlearning']['alpha']: {config['qlearning']['alpha']}")

    ## Init params 

    #alpha = config['qlearning']['alpha']
    #print(f"alpha: {alpha}")

  # execute = f"{method}_{algorithm}_{agent}"
    execute = f"{config['Method']}_{config['Algorithm']}_{config['Agent']}"
    print(f"execute: {execute}")


    if execute == 'train_qlearning_f1':
        train_qlearning_f1(config)
        




