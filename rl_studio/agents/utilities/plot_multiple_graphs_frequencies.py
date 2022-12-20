import pickle

import matplotlib as pltlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

RUNS = 100
max_episode_steps = 500

intensities = [0, 0.1, 0.2, 0.4, 0.6, 0.8]

yticks = []

def plot_intensities(ax, file_0, file_1, file_2, file_3, file_4, file_5, label, color):

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

    rewards_file = open(
        file_5,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_5 = ECDF(rewards)

    ninety_rewards = [1- ecdf_0(499), 1 - ecdf_1(499), 1 - ecdf_2(499), 1 - ecdf_3(499), 1 - ecdf_4(499), 1-  ecdf_5(499)]

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
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 20:03:12.387811__rewards_rsl-0_rpl-0.1_pi-7.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-22 00:18:21.530321__rewards_rsl-0_rpl-0.6_pi-1_init_0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-22 00:18:29.673944__rewards_rsl-0_rpl-0.8_pi-1_init_0.pkl",
                     "ppo",
                     "green")
    #DQN
    plot_intensities(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:16:16.550832__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-22 00:15:24.492958__rewards_rsl-0_rpl-0.1_pi-1_init_0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-22 00:16:01.716049__rewards_rsl-0_rpl-0.2_pi-1_init_0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-22 00:16:02.753516__rewards_rsl-0_rpl-0.4_pi-1_init_0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-22 00:16:08.270695__rewards_rsl-0_rpl-0.6_pi-1_init_0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-22 00:16:20.325513__rewards_rsl-0_rpl-0.8_pi-1_init_0.pkl",
                     "DQN",
                     "red")

    #MANUAL
    plot_intensities(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:16:16.550832__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:18:57.234844__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:19:00.746120__rewards_rsl-0_rpl-0.2_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:19:12.534704__rewards_rsl-0_rpl-0.4_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:19:18.254783__rewards_rsl-0_rpl-0.6_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:19:23.058775__rewards_rsl-0_rpl-0.8_pi-1.pkl",
                     "programmatic",
                     "blue")

    #QLEAN
    plot_intensities(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:16:16.550832__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-22 00:17:05.210740__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-22 00:17:25.785325__rewards_rsl-0_rpl-0.2_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-22 00:26:55.906779__rewards_rsl-0_rpl-0.5_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-22 00:27:30.106815__rewards_rsl-0_rpl-0.8_pi-1.pkl",
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
    plt.xlabel("frequency of perturbations with fixed intensity")
    plt.ylabel("percentage of successful episodes")
    plt.grid()
    plt.legend()

    plt.show()
