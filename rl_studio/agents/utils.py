from datetime import datetime, timedelta
import logging
import os
import pickle
from pprint import pformat
import time

import cv2
import numpy as np
import pandas as pd
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer

from rl_studio.agents.f1 import settings


class LoggingHandler:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)

        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        f_format: Formatter = logging.Formatter(
            "[%(levelname)s] - %(asctime)s, filename: %(filename)s, funcname: %(funcName)s, line: %(lineno)s\n messages ---->\n %(message)s"
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def get_logger(self):
        return self.logger


def save_times(checkpoints):
    file_name = "actions_"
    file_dump = open(
        "./logs/" + file_name + settings.actions_set + "_checkpoints.pkl", "wb"
    )
    pickle.dump(checkpoints, file_dump)


def render(env, episode):
    render_skip = 0
    render_interval = 50
    render_episodes = 10

    if (episode % render_interval == 0) and (episode != 0) and (episode > render_skip):
        env.render()
    elif (
        ((episode - render_episodes) % render_interval == 0)
        and (episode != 0)
        and (episode > render_skip)
        and (render_episodes < episode)
    ):
        env.render(close=True)


class Bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_messages(*args, **kwargs):

    print(f"\n\t{Bcolors.OKCYAN}====>\t{args[0]}:{Bcolors.ENDC}")
    for key, value in kwargs.items():
        print(f"{Bcolors.OKBLUE}[INFO] {key} = {value}{Bcolors.ENDC}")
    print("\n")


def print_dictionary(dic):
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(dic)
    print(highlight(pformat(dic), PythonLexer(), Terminal256Formatter()), end="")


def render_params(**kwargs):
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.zeros((400, 1800, 3), dtype="uint8")
    white = (255, 255, 255)
    i = 10
    for key, value in kwargs.items():
        cv2.putText(
            canvas,
            str(f"{key}: {value}"),
            (20, i + 25),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        i += 25

    cv2.imshow("Control Board", canvas)
    cv2.waitKey(100)


def save_dataframe_episodes(environment, outdir, aggr_ep_rewards):
    """
    We save info every certains epochs in a dataframe and .npy format to export or manage
    """
    os.makedirs(f"{outdir}", exist_ok=True)
    file_name = f"{time.strftime('%Y%m%d-%H%M%S')}__States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}"
    file_csv = f"{outdir}/{file_name}.csv"
    file_excel = f"{outdir}/{file_name}.xlsx"

    df = pd.DataFrame(aggr_ep_rewards)
    df.to_csv(file_csv, mode="a", index=False, header=None)
    df.to_excel(file_excel)
    file_npy = f"{outdir}/{file_name}.npy"
    np.save(file_npy, aggr_ep_rewards)
    return f"{file_name}.npy"


def save_best_episode(
    global_params,
    cumulated_reward,
    episode,
    step,
    start_time_epoch,
    reward,
    image_center,
):
    """
    save best episode in training
    """

    current_max_reward = cumulated_reward
    best_epoch = episode
    best_step = step
    best_epoch_training_time = datetime.now() - start_time_epoch
    # saving params to show
    # self.actions_rewards["episode"].append(episode)
    # self.actions_rewards["step"].append(step)
    # self.actions_rewards["reward"].append(reward)
    global_params.actions_rewards["episode"].append(episode)
    global_params.actions_rewards["step"].append(step)
    # For continuous actios
    # self.actions_rewards["v"].append(action[0][0])
    # self.actions_rewards["w"].append(action[0][1])
    global_params.actions_rewards["reward"].append(reward)
    global_params.actions_rewards["center"].append(image_center)

    return current_max_reward, best_epoch, best_step, best_epoch_training_time


def save_best_episode_dqn(
    global_params,
    cumulated_reward,
    episode,
    step,
    start_time_epoch,
    reward,
):
    """
    save best episode in training
    """

    current_max_reward = cumulated_reward
    best_epoch = episode
    best_step = step
    best_epoch_training_time = datetime.now() - start_time_epoch
    # saving params to show
    # self.actions_rewards["episode"].append(episode)
    # self.actions_rewards["step"].append(step)
    # self.actions_rewards["reward"].append(reward)
    global_params.best_current_epoch["best_epoch"].append(episode)
    global_params.best_current_epoch["best_step"].append(step)
    # For continuous actios
    # self.actions_rewards["v"].append(action[0][0])
    # self.actions_rewards["w"].append(action[0][1])
    global_params.best_current_epoch["highest_reward"].append(reward)
    global_params.best_current_epoch["best_epoch_training_time"].append(
        best_epoch_training_time
    )
    global_params.best_current_epoch["current_total_training_time"].append(
        start_time_epoch
    )

    return current_max_reward, best_epoch, best_step, best_epoch_training_time


def save_batch(episode, step, start_time_epoch, start_time, global_params, env_params):
    """
    save batch of n episodes
    """
    average_reward = sum(global_params.ep_rewards[-env_params.save_episodes :]) / len(
        global_params.ep_rewards[-env_params.save_episodes :]
    )
    min_reward = min(global_params.ep_rewards[-env_params.save_episodes :])
    max_reward = max(global_params.ep_rewards[-env_params.save_episodes :])

    global_params.aggr_ep_rewards["episode"].append(episode)
    global_params.aggr_ep_rewards["step"].append(step)
    global_params.aggr_ep_rewards["avg"].append(average_reward)
    global_params.aggr_ep_rewards["max"].append(max_reward)
    global_params.aggr_ep_rewards["min"].append(min_reward)
    global_params.aggr_ep_rewards["epoch_training_time"].append(
        (datetime.now() - start_time_epoch).total_seconds()
    )
    global_params.aggr_ep_rewards["total_training_time"].append(
        (datetime.now() - start_time).total_seconds()
    )

    return global_params.aggr_ep_rewards


def load_model(qlearn, file_name):
    """
    Qlearn old version
    """

    qlearn_file = open("./logs/qlearn_models/" + file_name)
    model = pickle.load(qlearn_file)

    qlearn.q = model
    qlearn.ALPHA = settings.algorithm_params["alpha"]
    qlearn.GAMMA = settings.algorithm_params["gamma"]
    qlearn.epsilon = settings.algorithm_params["epsilon"]

    print(f"\n\nMODEL LOADED. Number of (action, state): {len(model)}")
    print(f"    - Loading:    {file_name}")
    print(f"    - Model size: {len(qlearn.q)}")
    print(f"    - Action set: {settings.actions_set}")
    print(f"    - Epsilon:    {qlearn.epsilon}")
    print(f"    - Start:      {datetime.datetime.now()}")


def save_model(qlearn, current_time, states, states_counter, states_rewards):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.

    # Q TABLE
    base_file_name = "_act_set_{}_epsilon_{}".format(
        settings.actions_set, round(qlearn.epsilon, 2)
    )
    file_dump = open(
        "./logs/qlearn_models/1_" + current_time + base_file_name + "_QTABLE.pkl", "wb"
    )
    pickle.dump(qlearn.q, file_dump)
    # STATES COUNTER
    states_counter_file_name = base_file_name + "_STATES_COUNTER.pkl"
    file_dump = open(
        "./logs/qlearn_models/2_" + current_time + states_counter_file_name, "wb"
    )
    pickle.dump(states_counter, file_dump)
    # STATES CUMULATED REWARD
    states_cum_reward_file_name = base_file_name + "_STATES_CUM_REWARD.pkl"
    file_dump = open(
        "./logs/qlearn_models/3_" + current_time + states_cum_reward_file_name, "wb"
    )
    pickle.dump(states_rewards, file_dump)
    # STATES
    steps = base_file_name + "_STATES_STEPS.pkl"
    file_dump = open("./logs/qlearn_models/4_" + current_time + steps, "wb")
    pickle.dump(states, file_dump)
