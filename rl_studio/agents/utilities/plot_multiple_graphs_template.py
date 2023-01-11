import pickle

import matplotlib as pltlib
import matplotlib.pyplot as plt
import numpy as np

RUNS = 20000
max_episode_steps = 500

if __name__ == "__main__":
    pltlib.rcParams.update({'font.size': 15})

    fig, ax1 = plt.subplots()

    rewards_file = open(
        "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/training/2023-01-06 22:53:15.067116__rewards_rsl-0_rpl-0_pi-0.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.plot(range(RUNS), rewards, color='orange', label='ddpg')

    RUNS = 1000

    rewards_file = open(
        "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/training/2023-01-11 21:25:12.509340__rewards_rsl-0_rpl-0_pi-0.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.plot(range(RUNS), rewards, color='green', label='ppo_continuous')

    RUNS = 1000

    rewards_file = open(
        "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/training/2023-01-11 21:30:20.951490__rewards_rsl-0.2_rpl-0_pi-0.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.plot(range(RUNS), rewards, color='blue', label='ppo')
    #
    # rewards_file = open(
    #     "/rl_studio/logs/cartpole/old_datasets/training_with_frequencies/2022-10-20 22:59:30.164014__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    # rewards = pickle.load(rewards_file)
    # rewards = np.asarray(rewards)
    # ax1.plot(range(RUNS), rewards, color='black', label='trained with frequency= 0.3')

    plt.legend()
    plt.ylabel("steps")
    plt.xlabel("runs")
    plt.show()

