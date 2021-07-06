from datetime import datetime
import time
import sys
import argparse
from cprint import cprint
from functools import reduce

import gym
import numpy as np
import os
from icecream import ic

# next imports
# from agents.f1.settings
# from agents.f1.liveplot
# from agents.f1.utils
# from agents.f1.qlearn

# Importing app local files
import settings
import liveplot
import utils

#from algorithms.qlearn import QLearn
from algorithms.Tabulars.train_qlearning import train_qlearning
from algorithms.Tabulars.test_qlearning import test_qlearning
from algorithms.Approximate.DQN.train_dqn import train_dqn


ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')



if __name__ == '__main__':

    # ------------------- Parameter parsing from YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()

    #config, algorithm1, algorithm_hyperparams1, model1, actions1, gaz_pos1 = utils.read_config(args.config_file)
    config = utils.read_config(args.config_file)

    execute_algor = f"{config['Method']}_{config['Algorithm']}"
    #print(f"\n [RLStudio] -> execute ALGORITHM : {execute_algor}")
    ic(execute_algor)


    # ------------------- CREATE DIRS
    os.makedirs("logs", exist_ok=True)
    #os.makedirs("stats", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    #os.makedirs("tables", exist_ok=True)
    #os.makedirs("logs", exist_ok=True)
    


    if execute_algor == 'train_qlearning':
        train_qlearning(config)

    if execute_algor == 'test_qlearning':
        test_qlearning(config)        
        
    if execute_algor == 'train_dqn':
        train_dqn(config)    



