
import time
from tqdm import tqdm
from cprint import cprint
from functools import reduce
import os

import gym

import numpy as np

import settings
import liveplot
import utils
from icecream import ic
from datetime import datetime, timedelta

from algorithms.Tabulars.qlearn import QLearn

from gym_gazebo.envs.gazebo_env import *


ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')


def test_qlearning(config):

    '''
        - Take best training: Q Table
        - Execute n_test times


    '''


    #--------------------- Init QLearning Vars
    model = config['Model']

    # ------------------ Init env
    environment = {}
    environment['agent'] = config['Agent']
    environment['env'] = config['envs_params'][model]['env']
    environment['training_type'] = config['envs_params'][model]['training_type']
    environment['circuit_name'] = config['envs_params'][model]['circuit_name']
    environment['actions'] = settings.AVAILABLE_ACTIONS[settings.actions_set]
    environment['launch'] = config['envs_params'][model]['launch']
    environment['gaz_pos'] = settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']]
    environment['start_pose'] = [settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']][1][1], settings.GAZEBO_POSITIONS[config['envs_params'][model]['circuit_name']][1][2]]
    environment['alternate_pose'] = config['envs_params'][model]['alternate_pose']
    environment['estimated_steps'] = config['envs_params'][model]['estimated_steps']
    environment['sensor'] = config['envs_params'][model]['sensor']
    environment['rewards'] = config['Rewards']
    environment['ROS_MASTER_URI'] = config['ROS_MASTER_URI']
    environment['GAZEBO_MASTER_URI'] = config['GAZEBO_MASTER_URI']


    #print(f"\n [train_qlearning_f1] -> environment: {environment}")
    ic(environment)

    env = gym.make(environment["env"], **environment)

    #cprint.info(f"\n ---- [train_qlearning_f1] -> come back train_qlearn_f1 ------------")

    # ---------------- Init hyperparmas & Algorithm
    alpha = config['Hyperparams']['alpha']
    gamma = config['Hyperparams']['gamma']
    initial_epsilon = config['Hyperparams']['epsilon']
    epsilon = config['Hyperparams']['epsilon']
    epsilon_discount = config['Hyperparams']['epsilon_discount']
    total_episodes = config['Hyperparams']['total_episodes']
    estimated_steps = config['envs_params'][model]['estimated_steps']

    actions = range(env.action_space.n) # lo recibe de F1QlearnCameraEnv
    ic(actions)      


    '''
        LOAD qtable to tested.

    '''
    if config['load_qtable']:
        q_table = config['table_loaded_tested']
        qlearn = QLearn(actions=actions, alpha=alpha, gamma=gamma, epsilon=epsilon, q_table=q_table)


    '''
        START testing
    '''

    for episode in range(config["num_test"]):

        counter = 0
        done = False
        lap_completed = False

        cumulated_reward = 0
        epsilon = 0
        observation = env.reset()
        state = ''.join(map(str, observation))

        while True:
            action = qlearn.selectAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if done:
                print(f"episode reward: {cumulated_reward} in num_test: {episode}")
                break




