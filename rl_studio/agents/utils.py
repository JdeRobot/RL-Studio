import datetime
import os
import pickle
import time

import cv2
import numpy as np
import pandas as pd

from rl_studio.agents.f1 import settings


def load_model(qlearn, file_name):
    """
    Qlearn only
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
    """
    Qlearn only
    """
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

    print(f"\n\t{Bcolors.OKCYAN}====>\t{args[0]}:{Bcolors.ENDC}\n")
    for key, value in kwargs.items():
        print(f"\t{Bcolors.OKBLUE}[INFO] {key} = {value}{Bcolors.ENDC}")
    print("\n")


def render_params(**kwargs):
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.zeros((400, 400, 3), dtype="uint8")
    # blue = (255, 0, 0)
    # green = (0, 255, 0)
    # red = (0, 0, 255)
    white = (255, 255, 255)
    # white_darkness = (200, 200, 200)
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


def save_dataframe_episodes(environment, outdir, aggr_ep_rewards, actions_rewards=None):
    """
    We save info every certains epochs in a dataframe and .npy format to export or manage
    """
    os.makedirs(f"{outdir}", exist_ok=True)

    file_csv = f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{environment['env']}_States-{environment['states']}_Actions-{environment['actions']}_rewards-{environment['rewards']}.csv"
    file_excel = f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{environment['env']}_States-{environment['states']}_Actions-{environment['actions']}_rewards-{environment['rewards']}.xlsx"

    df = pd.DataFrame(aggr_ep_rewards)
    df.to_csv(file_csv, mode="a", index=False, header=None)
    df.to_excel(file_excel)

    if actions_rewards is not None:
        file_npy = f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{environment['env']}_States-{environment['states']}_Actions-{environment['actions']}_rewards-{environment['rewards']}.npy"
        np.save(file_npy, actions_rewards)


def save_model_qlearn(
    environment,
    outdir,
    qlearn,
    current_time,
    steps_epochs,
    states_counter,
    states_rewards,
    episode,
    step,
    epsilon,
):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.

    outdir_models = f"{outdir}_models"
    os.makedirs(f"{outdir_models}", exist_ok=True)

    # Q TABLE
    # base_file_name = "_actions_set:_{}_epsilon:_{}".format(settings.actions_set, round(qlearn.epsilon, 2))
    base_file_name = f"_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{states_rewards}"
    file_dump = open(
        f"{outdir_models}/1_" + current_time + base_file_name + "_QTABLE.pkl", "wb"
    )
    pickle.dump(qlearn.q, file_dump)

    # STATES COUNTER
    states_counter_file_name = base_file_name + "_STATES_COUNTER.pkl"
    file_dump = open(
        f"{outdir_models}/2_" + current_time + states_counter_file_name, "wb"
    )
    pickle.dump(states_counter, file_dump)

    # STATES CUMULATED REWARD
    states_cum_reward_file_name = base_file_name + "_STATES_CUM_REWARD.pkl"
    file_dump = open(
        f"{outdir_models}/3_" + current_time + states_cum_reward_file_name, "wb"
    )
    pickle.dump(states_rewards, file_dump)

    # STATES
    steps = base_file_name + "_STATES_STEPS.pkl"
    file_dump = open(f"{outdir_models}/4_" + current_time + steps, "wb")
    pickle.dump(steps_epochs, file_dump)
