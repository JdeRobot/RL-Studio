import pickle

import matplotlib as pltlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import os
import re

RUNS = 100
max_episode_steps = 500

min_pert_freq = 0.1

intensities = []

yticks = []


def plot_freq(ax, folder_path, color):
    # Use a regular expression to extract the part between "cartpole" and "inference"
    match = re.search(r'cartpole/(.*?)/inference', folder_path)
    if match:
        extracted_part = match.group(1)
        label = extracted_part
    else:
        label = "unknown"
        print("No match found")

    file_dict = {}
    ecdf_dict = {}

    # Iterate through all the files in the folder
    for file_name in os.listdir(folder_path):
        # Use a regular expression to extract the part between "pi" and "_in"
        match = re.search(r'rpl-(.*?)_pi', file_name)
        match2 = re.search(r'pi-(.*?)_in', file_name)
        match3 = re.search(r'init_(.*?).pkl', file_name)

        if match:
            extracted_part = float(match.group(1))

            if match2 and match3:
                extracted_part2 = float(match2.group(1))
                extracted_part3 = float(match3.group(1))
                if extracted_part == 0 and extracted_part2 == 0 and extracted_part3 == 0:
                    # Add the extracted part and filename to the list
                    rewards_file = open(folder_path+file_name, "rb")
                    rewards = pickle.load(rewards_file)
                    rewards = np.asarray(rewards)
                    ecdf = ECDF(rewards)
                    ecdf_dict[float(extracted_part)] = 1 - ecdf(499)
            if extracted_part > min_pert_freq:
                rewards_file = open(folder_path+file_name, "rb")
                rewards = pickle.load(rewards_file)
                rewards = np.asarray(rewards)
                ecdf = ECDF(rewards)
                ecdf_dict[float(extracted_part)] = 1 - ecdf(499)

    print(label)
    sorted_dict = dict(sorted(ecdf_dict.items(), key=lambda item: item[0]))
    print(sorted_dict)
    extracted_intensities = list(sorted_dict.keys())
    success_percentage = list(sorted_dict.values())

    yticks.extend(success_percentage)
    intensities.extend(extracted_intensities)

    ax.plot(extracted_intensities, success_percentage, color=color, label=label)

def cleanticks(ticks):
    clear_ticks = []
    element1 = ticks[0]
    clear_ticks.append(element1)
    for element2 in ticks:
        if element1 != element2 and abs(element1 - element2) > 0.02:
            clear_ticks.append(element2)
            element1 = element2
    return clear_ticks


if __name__ == "__main__":
    pltlib.rcParams.update({'font.size': 15})

    fig, ax1 = plt.subplots()

    # PPO
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/", "green")
    # DQN
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/", "red")
    # MANUAL
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/", "blue")
    # QLEAN
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/", "purple")
    # PPO CONTINUOUS
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/", "black")
    # DDPG
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/inference/", "brown")

    plt.xticks(intensities)
    yticks = np.array(yticks)
    flatten_ticks = yticks.flatten()
    clear_ticks = cleanticks(sorted(flatten_ticks, reverse=True))
    plt.yticks(clear_ticks)
    plt.setp(ax1.get_yticklabels(), horizontalalignment='right', fontsize='xx-small')
    plt.setp(ax1.get_xticklabels(), horizontalalignment='right', fontsize='x-small')
    plt.xlabel("intensity of perturbations with fixed frequency")
    plt.ylabel("percentage of successful episodes")
    plt.grid()
    plt.legend()

    plt.show()
