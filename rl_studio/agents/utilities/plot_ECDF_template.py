import pickle
import matplotlib as pltlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

RUNS = 100
max_episode_steps = 500

def plot_ecdf(sample, color, label):
    # fit a cdf
    ecdf = ECDF(sample)
    # get cumulative probability for values
    print('P(x<20): %.3f' % ecdf(20))
    print('P(x<40): %.3f' % ecdf(40))
    print('P(x<60): %.3f' % ecdf(60))
    # plot the cdf
    plt.plot(ecdf.x, ecdf.y,  color=color, label=label)

if __name__ == "__main__":

    pltlib.rcParams.update({'font.size': 15})

    rewards_file = open(
        "/rl_studio/logs/cartpole/old_datasets/dqn_analysis/training_with_frequencies/2022-10-20 23:05:26.343167__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    plot_ecdf(rewards, 'blue', 'trained with frequency = 0')

    rewards_file = open(
        "/rl_studio/logs/cartpole/old_datasets/dqn_analysis/training_with_frequencies/2022-10-20 23:00:41.230018__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    plot_ecdf(rewards, 'green', 'trained with frequency = 0.1')

    rewards_file = open(
        "/rl_studio/logs/cartpole/old_datasets/dqn_analysis/training_with_frequencies/2022-10-20 23:00:04.352224__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    plot_ecdf(rewards, 'orange', 'trained with frequency = 0.2')

    rewards_file = open(
        "/rl_studio/logs/cartpole/old_datasets/dqn_analysis/training_with_frequencies/2022-10-20 22:59:30.164014__rewards_rsl-0_rpl-0.2_pi-10.pkl", "rb")
    rewards = pickle.load(rewards_file)
    rewards = np.asarray(rewards)
    plot_ecdf(rewards, 'black', 'trained with frequency = 0.3')

    plt.legend()
    plt.ylabel("percentage of runs")
    plt.xlabel("steps")
    plt.show()

