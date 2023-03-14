import matplotlib as pltlib
import matplotlib.pyplot as plt
import numpy as np

def load_rewards(file_path):
    rewards = np.load(file_path, allow_pickle=True).item()
    return rewards


def plot_rewards(base_directory_name, file_name):
    pltlib.rcParams.update({'font.size': 15})
    fig, ax1 = plt.subplots()

    file_path = base_directory_name + "/" + file_name
    rewards = load_rewards(file_path)
    ax1.set_xscale('log')
    ax1.plot(range(len(rewards["avg"])), rewards["avg"], color='purple', label='ddpg')

    # fig.canvas.manager.full_screen_toggle()

    plt.legend()
    plt.ylabel("cun_reward")
    plt.xlabel("episode")
    # plt.show()

    base_path = '/home/ruben/Desktop/2020-phd-ruben-lucas/docs/assets/images/results_images/f1-follow-line/gazebo/ddpg/sp10/'
    ax1.figure.savefig(base_path + 'trainings.png')

if __name__ == "__main__":
    # Pass the file_name as a parameter when calling the script
    file_name = "20230722-135205_Circuit-simple_States-sp10_Actions-continuous_Rewards-followline_center.npy"
    plot_rewards(file_name)
