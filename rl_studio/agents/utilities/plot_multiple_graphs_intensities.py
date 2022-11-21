import pickle

import matplotlib as pltlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

RUNS = 100
max_episode_steps = 500

intensities = [0, 1, 7, 12, 18]

yticks = []

def plot_intensities(ax, file_0, file_1, file_2, file_3, file_4, label, color):

    rewards_file = open(
        file_0,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_0 = ECDF(rewards)

    rewards_file = open(
        file_1,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_1 = ECDF(rewards)

    rewards_file = open(
        file_2,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_2 = ECDF(rewards)

    rewards_file = open(
        file_3,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_3 = ECDF(rewards)

    rewards_file = open(
        file_4,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_4 = ECDF(rewards)
    ninety_rewards = [1- ecdf_0(499), 1 - ecdf_1(499), 1 - ecdf_2(499), 1 - ecdf_3(499), 1 - ecdf_4(499)]

    yticks.append(ninety_rewards)

    ax.plot(intensities, ninety_rewards, color=color, label=label)


def cleanticks(ticks):
    clear_ticks = []
    element1 = ticks[0]
    clear_ticks.append(element1)
    for element2 in ticks:
        if element1 != element2 and abs(element1 - element2) > 0.02:
            clear_ticks.append(element2)
    return clear_ticks

if __name__ == "__main__":
    pltlib.rcParams.update({'font.size': 15})

    fig, ax1 = plt.subplots()

    #PPO
    plot_intensities(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:16:16.550832__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 20:03:12.387811__rewards_rsl-0_rpl-0.1_pi-7.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 20:03:18.660196__rewards_rsl-0_rpl-0.1_pi-12.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 20:03:43.891839__rewards_rsl-0_rpl-0.1_pi-18.pkl",
                     "ppo",
                     "green")
    #DQN
    plot_intensities(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:16:16.550832__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-16 20:35:59.222709__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-16 20:36:33.282602__rewards_rsl-0_rpl-0.1_pi-7.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-16 20:36:51.443741__rewards_rsl-0_rpl-0.1_pi-12.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-16 20:37:33.130595__rewards_rsl-0_rpl-0.1_pi-18.pkl",
                     "DQN",
                     "red")

    #MANUAL
    plot_intensities(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:16:16.550832__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-16 20:40:06.485079__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-16 20:40:44.833057__rewards_rsl-0_rpl-0.1_pi-7.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-16 20:40:51.609087__rewards_rsl-0_rpl-0.1_pi-12.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-16 20:41:02.009116__rewards_rsl-0_rpl-0.1_pi-18.pkl",
                     "programmatic",
                     "blue")

    #QLEAN
    plot_intensities(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:16:16.550832__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:16:49.854808__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:17:27.826748__rewards_rsl-0_rpl-0.1_pi-7.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:17:27.826748__rewards_rsl-0_rpl-0.1_pi-7.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:17:27.826748__rewards_rsl-0_rpl-0.1_pi-7.pkl",
                     "QLearning",
                     "purple")

    plt.xticks(intensities)
    yticks = np.array(yticks)
    flatten_ticks = yticks.flatten()
    clear_ticks = cleanticks(flatten_ticks)
    plt.yticks(clear_ticks)
    plt.setp(ax1.get_yticklabels(), horizontalalignment='right', fontsize='xx-small')
    plt.setp(ax1.get_xticklabels(), horizontalalignment='right', fontsize='x-small')
    plt.xlabel("intensity of perturbations with fixed frequency")
    plt.ylabel("percentage of successful episodes")
    plt.grid()
    plt.legend()

    plt.show()
