import pickle

import matplotlib.pyplot as plt
import numpy as np

RUNS = 100
max_episode_steps = 500

if __name__ == "__main__":

    fig, ax1 = plt.subplots()

    rewards_file = open(
        "/rl_studio/logs/cartpole/old_datasets/training_with_frequencies/2022-10-20 23:05:26.343167__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.plot(range(RUNS), rewards, color='blue', label='trained with frequency = 0')

    rewards_file = open(
        "/rl_studio/logs/cartpole/old_datasets/training_with_frequencies/2022-10-20 23:00:41.230018__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.plot(range(RUNS), rewards, color='green', label='trained with frequency = 0.1')

    rewards_file = open(
        "/rl_studio/logs/cartpole/old_datasets/training_with_frequencies/2022-10-20 23:00:04.352224__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.plot(range(RUNS), rewards, color='orange', label='trained with frequency = 0.2')

    rewards_file = open(
        "/rl_studio/logs/cartpole/old_datasets/training_with_frequencies/2022-10-20 22:59:30.164014__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    ax1.plot(range(RUNS), rewards, color='black', label='trained with frequency= 0.3')

    plt.legend()
    plt.ylabel("steps")
    plt.xlabel("runs")
    plt.show()

