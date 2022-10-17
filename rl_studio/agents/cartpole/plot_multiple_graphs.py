import pickle

import matplotlib.pyplot as plt
import numpy as np

RUNS = 100
max_episode_steps = 500

if __name__ == "__main__":

    fig, ax1 = plt.subplots()

    rewards_file = open("", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    my_color = np.where(rewards == max_episode_steps, 'green', 'red')
    plt.scatter(range(RUNS), rewards, color=my_color, marker='x')
    ax1.plot(range(RUNS), rewards, color='blue', label='trained with ...')

    rewards_file = open("", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    my_color = np.where(rewards == max_episode_steps, 'green', 'red')
    plt.scatter(range(RUNS), rewards, color=my_color, marker='x')
    ax1.plot(range(RUNS), rewards, color='green', label='trained with ...')

    rewards_file = open("", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    my_color = np.where(rewards == max_episode_steps, 'green', 'red')
    plt.scatter(range(RUNS), rewards, color=my_color, marker='x')
    ax1.plot(range(RUNS), rewards, color='orange', label='trained with ...')

    rewards_file = open("", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    my_color = np.where(rewards == max_episode_steps, 'green', 'red')
    plt.scatter(range(RUNS), rewards, color=my_color, marker='x')
    ax1.plot(range(RUNS), rewards, color='black', label='trained with ...')

    plt.legend()
    plt.show()

