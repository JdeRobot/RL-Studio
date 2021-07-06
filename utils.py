import pickle
from datetime import datetime, timedelta
#from agents.f1 import settings
import settings

import os
import yaml
import csv
import time
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def read_config(yaml_file):
    '''
    read and parse YAML file with config
    '''
    with open(yaml_file, 'r') as stream:
        config_yaml = yaml.safe_load(stream)
    
    #algorithm = config_yaml['Algorithm']
    #model = config_yaml['Model']
    #actions = config_yaml['envs_params'][model]['actions'] 
    #gazebo_position = config_yaml['envs_params'][model]['gaz_pos']

    #return config_yaml, config_yaml['Algorithm'], config_yaml[algorithm], config_yaml['Model'], config_yaml[actions], config_yaml[gazebo_position]   
    return config_yaml


def save_stats_episodes(outdir, aggr_ep_rewards, config, episode):
    '''
            We save info of EPISODES (each 1 or n, which is defined in YMAL)
            in a dataframe to export or manage
    '''


    outdir_episode = f"{outdir}_{config['Model']}_STATS"
    os.makedirs(f"{outdir_episode}", exist_ok=True)

    file = open(f"{outdir_episode}/{config['Method']}_{config['Algorithm']}_{config['Agent']}_{config['Model']}_EPISODE_{episode}_{time.strftime('%Y%m%d-%H%M%S')}.csv", "a")
    writer = csv.writer(file)

    for key, value in aggr_ep_rewards.items():
        writer.writerow([key, value])

    file.close()    



def draw_rewards(aggr_ep_rewards, config, episode):
    '''
        Plot Rewards
    '''
    outdir_images = f"{config['Dirspace']}/images/{config['Method']}_{config['Algorithm']}_{config['Agent']}_{config['Model']}"
    os.makedirs(f"{outdir_images}", exist_ok=True)
    #outdir_images = f"{outdir_images}/{config['Method']}_{config['Algorithm']}_{config['Agent']}_{config['Model']}"


    plt.plot(aggr_ep_rewards['episode'], aggr_ep_rewards['avg'], label="average rewards")
    plt.plot(aggr_ep_rewards['episode'], aggr_ep_rewards['max'], label="max rewards")
    plt.plot(aggr_ep_rewards['episode'], aggr_ep_rewards['min'], label="min rewards")
    plt.title(f"{config['Method']}_{config['Algorithm']}_{config['Agent']}_{config['Model']} with Learning Rate {config['Hyperparams']['alpha']} and Discount Rate {config['Hyperparams']['gamma']}")

    plt.suptitle([config['Hyperparams']['alpha'], config['Hyperparams']['alpha']])
    plt.legend(loc=0) #loc=0 best place
    plt.grid(True)

    #os.makedirs("Images", exist_ok = True)            
    plt.savefig(f"{outdir_images}/{config['Method']}_{config['Algorithm']}_{config['Agent']}_{config['Model']}_{config['Hyperparams']['alpha']}_{config['Hyperparams']['gamma']}-{episode}-{time.strftime('%Y%m%d-%H%M%S')}.png", bbox_inches='tight')
    plt.clf()


def save_tables_npy_rewards(outdir, qlearn, config, episode):

    outdir_tables = f"{outdir}_tables"
    os.makedirs(f"{outdir_tables}", exist_ok=True)
    #os.makedirs("Tables", exist_ok = True)
                #np.save(f"Tables/{type(agent).__name__}-{LEARNING_RATE}-{DISCOUNT}-{episode}-qtable.npy", q_table)
    np.save(f"{outdir_tables}/{config['Method']}_{config['Algorithm']}_{config['Agent']}_{config['Model']}_EPISODE_{episode}_{time.strftime('%Y%m%d-%H%M%S')}-qtable.npy", qlearn.q)
       

def load_model(outdir, qlearn, file_name, config):

    '''
        It is used for PICKLES files
    '''

    qlearn_file = open(f"{outdir}_models/{file_name}")
    model = pickle.load(qlearn_file)

    qlearn.q = model
    #qlearn.alpha = settings.algorithm_params["alpha"]
    qlearn.alpha = config["Hyperparams"]['alpha']
    qlearn.gamma = config["Hyperparams"]['gamma']
    qlearn.epsilon = config["Hyperparams"]['epsilon']
    #qlearn.gamma = settings.algorithm_params["gamma"]
    #qlearn.epsilon = settings.algorithm_params["epsilon"]
    # highest_reward = settings.algorithm_params["highest_reward"]

    print(f"\n\nMODEL LOADED. Number of (action, state): {len(model)}")
    print(f"    - Loading:    {file_name}")
    print(f"    - Model size: {len(qlearn.q)}")
    print(f"    - Action set: {settings.actions_set}")
    print(f"    - Epsilon:    {qlearn.epsilon}")
    print(f"    - Start:      {datetime.now()}")


def save_model(outdir, qlearn, current_time, states, states_counter, states_rewards):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.

    outdir_models = f"{outdir}_models"
    os.makedirs(f"{outdir_models}", exist_ok=True)

    # Q TABLE
    '''
    Q Table
    '''
    base_file_name = "_actions_set:_{}_epsilon:_{}".format(settings.actions_set, round(qlearn.epsilon, 2))
    file_dump = open(f"{outdir_models}/1_" + current_time + base_file_name + '_QTABLE.pkl', 'wb')
    pickle.dump(qlearn.q, file_dump)
    
    # STATES COUNTER
    '''
    count the STATES the agent were
    '''
    states_counter_file_name = base_file_name + "_STATES_COUNTER.pkl"
    file_dump = open(f"{outdir_models}/2_" + current_time + states_counter_file_name, 'wb')
    pickle.dump(states_counter, file_dump)
    
    # STATES CUMULATED REWARD
    '''
    reward in each state
    '''
    states_cum_reward_file_name = base_file_name + "_STATES_CUM_REWARD.pkl"
    file_dump = open(f"{outdir_models}/3_" + current_time + states_cum_reward_file_name, 'wb')
    pickle.dump(states_rewards, file_dump)
    
    # STATES
    '''
    episodes
    '''
    steps = base_file_name + "_STATES_STEPS.pkl"
    file_dump = open(f"{outdir_models}/4_" + current_time + steps, 'wb')
    pickle.dump(states, file_dump)


def save_times(checkpoints):
    file_name = "actions_"
    file_dump = open("./logs/" + file_name + settings.actions_set + '_checkpoints.pkl', 'wb')
    pickle.dump(checkpoints, file_dump)

def render(env, episode):
    render_skip = 0
    render_interval = 50
    render_episodes = 10

    if (episode % render_interval == 0) and (episode != 0) and (episode > render_skip):
        env.render()
    elif ((episode - render_episodes) % render_interval == 0) and (episode != 0) and (episode > render_skip) and \
            (render_episodes < episode):
        env.render(close=True)
