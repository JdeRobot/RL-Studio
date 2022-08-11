import datetime
import pickle
import os

import cv2
import numpy as np
import pandas as pd

from rl_studio.agents.f1 import settings


def load_model(qlearn, file_name):

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
    canvas = np.zeros((300, 300, 3), dtype="uint8")
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


def save_agent_physics(environment, outdir, physics, current_time):
    """ """

    outdir_episode = f"{outdir}_stats"
    os.makedirs(f"{outdir_episode}", exist_ok=True)

    file_npy = f"{outdir_episode}/{current_time}_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_rewards-{environment['reward_function']}.npy"

    np.save(file_npy, physics)


def save_stats_episodes(environment, outdir, aggr_ep_rewards, current_time):
    """
    We save info of EPISODES in a dataframe to export or manage
    """

    outdir_episode = f"{outdir}_stats"
    os.makedirs(f"{outdir_episode}", exist_ok=True)

    file_csv = f"{outdir_episode}/{current_time}_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_rewards-{environment['reward_function']}.csv"
    file_excel = f"{outdir_episode}/{current_time}_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_rewards-{environment['reward_function']}.xlsx"

    df = pd.DataFrame(aggr_ep_rewards)
    df.to_csv(file_csv, mode="a", index=False, header=None)
    df.to_excel(file_excel)
