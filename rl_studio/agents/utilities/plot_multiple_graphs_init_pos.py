import pickle

import matplotlib as pltlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from statistics import mean

RUNS = 100
max_episode_steps = 500

init_pos = [0, 0.1, 0.25, 0.3, 0.45, 0.5, 0.55, 0.6]

yticks = []

def plot_init_pos(ax, file_00, file_0, file_1, file_2, file_3, file_4, file_5, file_6, label, color):

    rewards_file = open(
        file_00,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_00 = mean(rewards)

    rewards_file = open(
        file_0,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_0 = mean(rewards)

    rewards_file = open(
        file_1,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_1 = mean(rewards)

    rewards_file = open(
        file_2,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_2 = mean(rewards)

    rewards_file = open(
        file_3,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_3 = mean(rewards)

    rewards_file = open(
        file_4,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_4 = mean(rewards)

    rewards_file = open(
        file_5,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_5 = mean(rewards)

    rewards_file = open(
        file_6,
        "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ecdf_6 = mean(rewards)

    ninety_rewards = [ecdf_00, ecdf_0, ecdf_1, ecdf_2, ecdf_3, ecdf_4, ecdf_5, ecdf_6]

    yticks.append(ninety_rewards)

    ax.plot(init_pos, ninety_rewards, color=color, label=label)


def cleanticks(ticks):
    clear_ticks = []
    ticks.sort();
    element1 = ticks[0]
    print(ticks)
    clear_ticks.append(element1)
    for index in range(len(ticks)):
        element2 = ticks[index]
        if element1 != element2 and abs(element1 - element2) > 40:
            print(element1)
            print(element2)
            clear_ticks.append(element2)
            element1 = element2
    return clear_ticks

if __name__ == "__main__":
    pltlib.rcParams.update({'font.size': 15})
    pltlib.rcParams.update({'font.size': 15})

    fig, ax1 = plt.subplots()

    #PPO
    plot_init_pos(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-22 00:04:00.543207__rewards_rsl-0_rpl-0_pi-0_init_0.5.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2023-01-10 23:35:22.217242__rewards_rsl-0_rpl-0_pi-0_init_0.55.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2023-01-10 23:35:33.741613__rewards_rsl-0_rpl-0_pi-0_init_0.6.pkl",
                     "ppo",
                     "green")
    #DQN
    plot_init_pos(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-21 23:51:45.523954__rewards_rsl-0_rpl-0_pi-0_init_0.2.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-21 23:51:45.523954__rewards_rsl-0_rpl-0_pi-0_init_0.2.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-22 00:36:30.175255__rewards_rsl-0_rpl-0_pi-0_init_0.3.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2022-11-21 23:52:08.387862__rewards_rsl-0_rpl-0_pi-0_init_0.5.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2023-01-10 23:52:54.459779__rewards_rsl-0_rpl-0_pi-0_init_0.5.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2023-01-10 23:43:20.714748__rewards_rsl-0_rpl-0_pi-0_init_0.55.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/2023-01-10 23:43:29.278092__rewards_rsl-0_rpl-0_pi-0_init_0.6.pkl",
                     "DQN",
                     "red")

    #MANUAL
    plot_init_pos(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:32:26.534755__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:32:26.534755__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:31:51.862801__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:32:09.926731__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2023-01-10 23:50:13.792319__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2022-11-22 00:32:45.402747__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2023-01-10 23:47:22.244262__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/2023-01-10 23:47:29.228261__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "programmatic",
                     "blue")

    #QLEAN
    plot_init_pos(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:30:57.818809__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-21 23:30:57.818809__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-22 00:00:02.418770__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2022-11-22 00:30:10.278754__rewards_rsl-0_rpl-0_pi-0.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2023-01-10 23:56:17.160288__rewards_rsl-0_rpl-1_pi-1_in-0.55.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2023-01-10 23:55:56.928313__rewards_rsl-0_rpl-1_pi-1_in-0.6.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2023-01-10 23:43:47.116307__rewards_rsl-0_rpl-1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/2023-01-10 23:43:53.900264__rewards_rsl-0_rpl-1_pi-1.pkl",
                     "QLearning",
                     "purple")

    plot_init_pos(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/2023-01-10 23:59:08.498690__rewards_rsl-0_rpl-0_pi-0_init_0.45.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/2023-01-10 23:59:08.498690__rewards_rsl-0_rpl-0_pi-0_init_0.45.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/2023-01-10 23:59:08.498690__rewards_rsl-0_rpl-0_pi-0_init_0.45.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/2023-01-10 23:59:08.498690__rewards_rsl-0_rpl-0_pi-0_init_0.45.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/2023-01-10 23:59:08.498690__rewards_rsl-0_rpl-0_pi-0_init_0.45.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/2023-01-10 23:58:28.488336__rewards_rsl-0_rpl-0_pi-0_init_0.5.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/2023-01-10 23:34:45.195425__rewards_rsl-0_rpl-0_pi-0_init_0.55.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/2023-01-10 23:34:53.943778__rewards_rsl-0_rpl-0_pi-0_init_0.6.pkl",
                     "PPO_Continuous",
                     "black")

    plot_init_pos(ax1,
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/2022-11-16 19:58:16.130830__rewards_rsl-0_rpl-0.1_pi-1.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/inference/2023-01-10 23:33:31.053936__rewards_rsl-0_rpl-0_pi-0_init_0.5.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/inference/2023-01-10 23:34:07.621209__rewards_rsl-0_rpl-0_pi-0_init_0.55.pkl",
                     "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/inference/2023-01-10 23:34:17.155157__rewards_rsl-0_rpl-0_pi-0_init_0.6.pkl",
                     "DDPG",
                     "grey")

    plt.xticks(init_pos)
    yticks = np.array(yticks)
    flatten_ticks = yticks.flatten()
    clear_ticks = cleanticks(flatten_ticks)
    plt.yticks(clear_ticks)
    plt.setp(ax1.get_yticklabels(), horizontalalignment='right', fontsize='xx-small')
    plt.setp(ax1.get_xticklabels(), horizontalalignment='right', fontsize='x-small')
    plt.xlabel("initial angle with no perturbations")
    plt.ylabel("steps per episode in average")
    plt.grid()
    plt.legend()

    plt.show()
