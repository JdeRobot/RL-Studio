import pickle

import matplotlib as pltlib
import matplotlib.pyplot as plt
import numpy as np

RUNS = 20000
max_episode_steps = 500

if __name__ == "__main__":
    pltlib.rcParams.update({'font.size': 15})

    fig, ax1 = plt.subplots()

    RUNS = 4000000

    rewards_file = open(
        "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/training/2023-01-20 18:41:16.873044__rewards_.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.set_xscale('log')
    ax1.plot(range(RUNS), rewards, color='purple', label='qlearning')

    RUNS = 20000

    rewards_file = open(
        "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/training/2023-01-20 02:50:09.991537__rewards_rsl-0_rpl-0_pi-1.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.set_xscale('log')
    ax1.plot(range(RUNS), rewards, color='pink', label='dqn')

    RUNS = 10000

    rewards_file = open(
        "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/training/2023-01-14 03:15:07.136008__rewards_rsl-0_rpl-0.1_pi-1.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.set_xscale('log')
    ax1.plot(range(RUNS), rewards, color='brown', label='ddpg')

    RUNS = 1000

    rewards_file = open(
        "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/training/2023-01-11 21:25:12.509340__rewards_rsl-0_rpl-0_pi-0.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.set_xscale('log')
    ax1.plot(range(RUNS), rewards, color='black', label='ppo_continuous')

    RUNS = 1000

    rewards_file = open(
        "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/training/2023-01-11 21:30:20.951490__rewards_rsl-0.2_rpl-0_pi-0.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.set_xscale('log')
    ax1.plot(range(RUNS), rewards, color='green', label='ppo')
    fig.canvas.manager.full_screen_toggle()

    plt.legend()
    plt.ylabel("steps")
    plt.xlabel("runs")
    plt.show()

    base_path = '/home/ruben/Desktop/2020-phd-ruben-lucas/docs/assets/images/results_images/cartpole/solidityExperiments/refinement/refinementOfRefinement/'
    ax1.figure.savefig(base_path + 'trainings.png')

