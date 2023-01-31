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
min_pert_intensity = 1

frequencies = []
intensities_dev = []

yticks = []
yticks_dev = []

def plot_freq(ax, folder_path, color, boxplot=False):
    # Use a regular expression to extract the part between "cartpole" and "inference"
    match = re.search(r'cartpole/(.*?)/inference', folder_path)
    if match:
        extracted_part = match.group(1)
        label = extracted_part
    else:
        label = "unknown"
        print("No match found")

    ecdf_dict = {}
    rewards_dict = {}

    # Iterate through all the files in the folder
    for file_name in os.listdir(folder_path):
        # Use a regular expression to extract the part between "pi" and "_in"
        match = re.search(r'rpl-(.*?)_pi', file_name)
        match2 = re.search(r'pi-(.*?)_in', file_name)
        match3 = re.search(r'init_(.*?).pkl', file_name)

        if "rewards" in file_name and match and match2 and match3:
            extracted_part = float(match.group(1))
            extracted_part2 = float(match2.group(1))
            extracted_part3 = float(match3.group(1))
            if extracted_part == 0 and extracted_part2 == 0 and extracted_part3 == 0\
                    or extracted_part > min_pert_freq\
                    or extracted_part == min_pert_freq and extracted_part2 == min_pert_intensity:
                # Add the extracted part and filename to the list
                rewards_file = open(folder_path+file_name, "rb")
                rewards = pickle.load(rewards_file)
                rewards = np.asarray(rewards)
                ecdf = ECDF(rewards)
                ecdf_dict[float(extracted_part)] = 1 - ecdf(499)
                rewards_dict[float(extracted_part)] = rewards

    print(label)
    sorted_ecdf_dict = dict(sorted(ecdf_dict.items(), key=lambda item: item[0]))
    print(sorted_ecdf_dict)
    extracted_frequencies = list(sorted_ecdf_dict.keys())

    frequencies.extend(extracted_frequencies)

    if boxplot:
        sorted_rewards_dict = dict(sorted(rewards_dict.items(), key=lambda item: item[0]))
        sorted_rewards = list(sorted_rewards_dict.values())

        if len(sorted_rewards) == 0:
            return

        ax.boxplot(sorted_rewards, positions= [ np.round(x, 2) for x in extracted_frequencies ], widths=0.03,
                   flierprops={'marker': 'o', 'markersize': 2})
        ax.legend([label])
        return

    success_percentage = list(sorted_ecdf_dict.values())

    yticks.extend(success_percentage)
    frequencies.extend(extracted_frequencies)

    ax.plot(extracted_frequencies, success_percentage, color=color, label=label)



def plot_deviations(ax, folder_path, color):
    # Use a regular expression to extract the part between "cartpole" and "inference"
    match = re.search(r'cartpole/(.*?)/inference', folder_path)
    if match:
        extracted_part = match.group(1)
        label = extracted_part
    else:
        label = "unknown"
        print("No match found")

    deviations_list = {}

    # Iterate through all the files in the folder
    for file_name in os.listdir(folder_path):
        # Use a regular expression to extract the part between "pi" and "_in"
        match = re.search(r'rpl-(.*?)_pi', file_name)
        match2 = re.search(r'pi-(.*?)_in', file_name)
        match3 = re.search(r'init_(.*?).pkl', file_name)

        if "states" in file_name and match and match2 and match3:
            extracted_part = float(match.group(1))
            extracted_part2 = float(match2.group(1))
            extracted_part3 = float(match3.group(1))

            if extracted_part == 0 and extracted_part2 == 0 and extracted_part3 == 0\
                    or extracted_part > min_pert_freq\
                    or extracted_part == min_pert_freq and extracted_part2 == min_pert_intensity:
                # Add the extracted part and filename to the list
                states_file = open(folder_path + file_name, "rb")
                states = pickle.load(states_file)
                states_array = [np.asarray(step) for runs in states for step in runs]
                if label == 'qlearning':
                    states_array = [(angle-50)/119.61722488 for angle in states_array]
                absolute_arr = np.abs(states_array)
                absolute_mean = np.mean(absolute_arr)
                deviations_list[float(extracted_part) / 10] = absolute_mean

    print(label)
    sorted_deviations_dict = dict(sorted(deviations_list.items(), key=lambda item: item[0]))
    print(sorted_deviations_dict)
    extracted_intensities = list(sorted_deviations_dict.keys())

    intensities_dev.extend(extracted_intensities)

    sorted_deviations = list(sorted_deviations_dict.values())

    yticks_dev.extend(sorted_deviations)
    ax.plot(extracted_intensities, sorted_deviations, color=color, label=label)

def cleanticks(ticks):
    clear_ticks = []
    element1 = ticks[0]
    clear_ticks.append(element1)
    for element2 in ticks:
        if element1 != element2 and abs(element1 - element2) > 0.02:
            clear_ticks.append(element2)
            element1 = element2
    return clear_ticks


# def configure_frequencies_graph(ax1, clear_ticks, frequencies):
def configure_frequencies_graph(ax1, frequencies):

    # ax1.set_yticks(clear_ticks)
    y_ticks = np.linspace(0, 1, 21)
    ax1.set_yticks(y_ticks)
    ax1.set_xticks(frequencies)

    yticklabels = ax1.get_yticklabels()
    for yticklabel in yticklabels:
        yticklabel.set_horizontalalignment('right')
        yticklabel.set_fontsize('xx-small')

    xticks = ax1.get_xticklabels()
    for xtick in xticks:
        xtick.set_horizontalalignment('right')
        xtick.set_fontsize('xx-small')
    ax1.grid()
    ax1.legend()

    ax1.set_xlabel("frequency of perturbations with fixed intensity")
    ax1.set_ylabel("percentage of successful episodes")

def configure_boxplot_graph(ax1, frequencies):
    boxplot_y = np.linspace(0, 500, 26)
    ax1.set_yticks(boxplot_y)
    ax1.set_xticks(frequencies)
    ax1.set_xlim(frequencies[0]-0.1, frequencies[len(frequencies)-1]+0.1)
    ax1.grid()
    ax1.set_xlabel("frequency of perturbations with fixed intensity")
    ax1.set_ylabel("number of steps")

# def configure_intensities_graph(ax1, clear_ticks, intensities):
def configure_deviation_graph(ax1, intensities):
    # ax1.set_yticks(clear_ticks)
    y_ticks_dev = np.linspace(0, 1, 21)
    ax1.set_yticks(y_ticks_dev)
    ax1.set_xticks(intensities)
    ax1.set_ylim(0, 0.1)

    yticklabels = ax1.get_yticklabels()
    for yticklabel in yticklabels:
        yticklabel.set_horizontalalignment('right')
        yticklabel.set_fontsize('xx-small')

    xticks = ax1.get_xticklabels()
    for xtick in xticks:
        xtick.set_horizontalalignment('right')
        xtick.set_fontsize('xx-small')
    ax1.grid()
    ax1.legend()

    ax1.set_xlabel("frequency of perturbations with fixed intensity")
    ax1.set_ylabel("average pole angle")

if __name__ == "__main__":
    pltlib.rcParams.update({'font.size': 15})

    fig, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4= plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    fig7, ax7 = plt.subplots()
    fig8, ax8 = plt.subplots()

    # PPO
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/", "green")
    plot_freq(ax2, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/", "green", True)
    # DQN
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/", "red")
    plot_freq(ax3, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/", "red", True)
    # MANUAL
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/", "blue")
    plot_freq(ax4, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/", "blue", True)
    # QLEAN
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/", "purple")
    plot_freq(ax5, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/", "purple", True)
    # PPO CONTINUOUS
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/", "black")
    plot_freq(ax6, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/", "black", True)
    # DDPG
    plot_freq(ax1, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/inference/", "brown")
    plot_freq(ax7, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/inference/", "brown", True)

    # Deviations
    plot_deviations(ax8, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo/inference/", "green")
    plot_deviations(ax8, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/dqn/inference/", "red")
    plot_deviations(ax8, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/no_rl/inference/", "blue")
    plot_deviations(ax8, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/qlearning/inference/", "purple")
    plot_deviations(ax8, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ppo_continuous/inference/", "black")
    plot_deviations(ax8, "/home/ruben/Desktop/my-RL-Studio/rl_studio/logs/cartpole/ddpg/inference/", "brown")


    configure_frequencies_graph(ax1, frequencies)
    configure_boxplot_graph(ax2, frequencies)
    configure_boxplot_graph(ax3, frequencies)
    configure_boxplot_graph(ax4, frequencies)
    configure_boxplot_graph(ax5, frequencies)
    configure_boxplot_graph(ax6, frequencies)
    configure_boxplot_graph(ax7, frequencies)
    configure_deviation_graph(ax8, intensities_dev)

    fig.canvas.manager.full_screen_toggle()
    fig2.canvas.manager.full_screen_toggle()
    fig3.canvas.manager.full_screen_toggle()
    fig4.canvas.manager.full_screen_toggle()
    fig5.canvas.manager.full_screen_toggle()
    fig6.canvas.manager.full_screen_toggle()
    fig7.canvas.manager.full_screen_toggle()
    fig8.canvas.manager.full_screen_toggle()

    plt.show()
    base_path = '/home/ruben/Desktop/2020-phd-ruben-lucas/docs/assets/images/results_images/cartpole/solidityExperiments/refinement/refinementOfRefinement/frequency/'
    ax1.figure.savefig(base_path + 'comparison.png')
    ax2.figure.savefig(base_path + 'ppo.png')
    ax3.figure.savefig(base_path + 'dqn.png')
    ax4.figure.savefig(base_path + 'no_rl.png')
    ax5.figure.savefig(base_path + 'qlearning.png')
    ax6.figure.savefig(base_path + 'qlearning.png')
    ax7.figure.savefig(base_path + 'ddpg.png')
    ax8.figure.savefig(base_path + 'deviations.png')
