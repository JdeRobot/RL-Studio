import pickle

import numpy as np
import matplotlib.pyplot as plt
from markdownTable import markdownTable

# How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded
LEARNING_RATE = 0.1
# Between 0 and 1, mesue of how much we carre about future reward over immedate reward
DISCOUNT = 0.95
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 10000 // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def save_model_qlearn(qlearn, current_time, avg):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.

    # TODO The paths are not relative to the agents folder
    # Q TABLE
    base_file_name = "_epsilon_{}".format(round(qlearn.epsilon, 3))
    file_dump = open(
        "./logs/cartpole/qlearning/checkpoints/" + current_time + base_file_name + "_QTABLE_avg_ " +  str(avg) + ".pkl",
        "wb"
    )
    pickle.dump(qlearn.q, file_dump)


def params_to_markdown_list(dictionary):
    md_list = []
    for item in dictionary:
        md_list.append({"parameter": item, "value": dictionary.get(item)})
    return md_list

def save_metadata(algorithm, current_time, params):
    metadata = open("./logs/cartpole/" + algorithm + "/checkpoints/" + current_time + "_metadata.md",
                    "a")
    metadata.write("AGENT PARAMETERS\n")
    metadata.write(markdownTable(params_to_markdown_list(params.get("agent"))).setParams(row_sep='always').getMarkdown())
    metadata.write("\n```\n\nSETTINGS PARAMETERS\n")
    metadata.write(markdownTable(params_to_markdown_list(params.get("settings"))).setParams(row_sep='always').getMarkdown())
    metadata.write("\n```\n\nENVIRONMENT PARAMETERS\n")
    metadata.write(markdownTable(params_to_markdown_list(params.get("environments"))).setParams(row_sep='always').getMarkdown())
    metadata.write("\n```\n\nALGORITHM PARAMETERS\n")
    metadata.write(markdownTable(params_to_markdown_list(params.get("algorithm"))).setParams(row_sep='always').getMarkdown())
    metadata.close()
def save_dqn_model(dqn, current_time, average, params):
    base_file_name = "_epsilon_{}".format(round(epsilon, 2))
    file_dump = open(
        "./logs/cartpole/dqn/checkpoints/" + current_time + base_file_name + "_DQN_WEIGHTS_avg_" + str(
            average) + ".pkl",
        "wb",
    )
    pickle.dump(dqn.q_net, file_dump)
    file_dump.close()


def save_ppo_model(actor, current_time, average, params):
    file_dump = open(
        "./logs/cartpole/ppo/checkpoints/" + current_time + "_actor_avg_" + str(
            average) + ".pkl",
        "wb",
    )
    pickle.dump(actor.model, file_dump)
    file_dump.close()


def save_actions_qlearn(actions, start_time, params):
    file_dump = open("./logs/cartpole/qlearning/checkpoints/actions_set_" + start_time, "wb")
    pickle.dump(actions, file_dump)
    file_dump.close()


# Create bins and Q table
def create_bins_and_q_table(env, number_angle_bins, number_pos_bins):
    # env.observation_space.high
    # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
    # env.observation_space.low
    # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

    # remove hard coded Values when I know how to

    obsSpaceSize = len(env.observation_space.high)

    # Get the size of each bucket
    bins = [
        np.linspace(-4.8, 4.8, number_pos_bins),
        np.linspace(-4, 4, number_pos_bins),
        np.linspace(-0.418, 0.418, number_angle_bins),
        np.linspace(-4, 4, number_angle_bins),
    ]

    qTable = np.random.uniform(
        low=-2, high=0, size=([number_pos_bins] * int(obsSpaceSize/2) + [number_angle_bins] * int(obsSpaceSize/2) + [env.action_space.n])
    )

    return bins, obsSpaceSize, qTable


# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, bins, obsSpaceSize):
    stateIndex = []
    for i in range(obsSpaceSize):
        stateIndex.append(
            np.digitize(state[i], bins[i]) - 1
        )  # -1 will turn bin into index
    return tuple(stateIndex)


def extract(lst, pos):
    return [item[pos] for item in lst]


def plot_detail_random_start_level_monitoring(unsuccessful_episodes_count, unsuccessful_initial_states,
                                              unsuccess_rewards, success_rewards, successful_initial_states,
                                              RUNS, random_start_level):
    figure, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    ax1.plot(range(unsuccessful_episodes_count), extract(unsuccessful_initial_states, 0))
    ax1.set(title="FAILURES: initial states with random level = " + str(random_start_level),
            ylabel="Cart Position")
    ax2.plot(range(unsuccessful_episodes_count), extract(unsuccessful_initial_states, 1))
    ax2.set(ylabel="Cart Velocity")
    ax3.plot(range(unsuccessful_episodes_count), extract(unsuccessful_initial_states, 2))
    ax3.set(ylabel="Pole Angle")
    ax4.plot(range(unsuccessful_episodes_count), extract(unsuccessful_initial_states, 3))
    ax4.set(ylabel="Pole Angular Velocity")
    ax5.plot(range(unsuccessful_episodes_count), unsuccess_rewards)
    ax5.set(ylabel="Rewards")

    figure2, (ax6, ax7, ax8, ax9, ax10) = plt.subplots(5)
    ax6.plot(range(RUNS - unsuccessful_episodes_count), extract(successful_initial_states, 0))
    ax6.set(title="SUCCESS: initial states with random level = " + str(random_start_level),
            ylabel="Cart Position")
    ax7.plot(range(RUNS - unsuccessful_episodes_count), extract(successful_initial_states, 1))
    ax7.set(ylabel="Cart Velocity")
    ax8.plot(range(RUNS - unsuccessful_episodes_count), extract(successful_initial_states, 2))
    ax8.set(ylabel="Pole Angle")
    ax9.plot(range(RUNS - unsuccessful_episodes_count), extract(successful_initial_states, 3))
    ax9.set(ylabel="Pole Angular Velocity")
    ax10.plot(range(RUNS - unsuccessful_episodes_count), success_rewards)
    ax10.set(ylabel="Rewards")


def plot_detail_random_perturbations_monitoring(unsuccessful_episodes_count, success_perturbations_in_twenty,
                                                success_max_perturbations_in_twenty_run, success_rewards,
                                                unsuccess_perturbations_in_twenty,
                                                unsuccess_max_perturbations_in_twenty_run, unsuccess_rewards,
                                                RUNS, RANDOM_PERTURBATIONS_LEVEL, PERTURBATIONS_INTENSITY):
    figure3, (ax11, ax12, ax13) = plt.subplots(3)
    ax11.plot(range(RUNS - unsuccessful_episodes_count), success_perturbations_in_twenty)
    ax11.set(title="SUCCESS: perturbation level "
                   "= " + str(RANDOM_PERTURBATIONS_LEVEL) + " and intensity = " + str(PERTURBATIONS_INTENSITY),
             ylabel="max number of perturbations in twenty consecutive steps")
    ax12.plot(range(RUNS - unsuccessful_episodes_count), success_max_perturbations_in_twenty_run)
    ax12.set(ylabel="in which step")
    ax13.plot(range(RUNS - unsuccessful_episodes_count), success_rewards)
    ax13.set(ylabel="Rewards")

    figure4, (ax14, ax15, ax16) = plt.subplots(3)
    ax14.plot(range(unsuccessful_episodes_count), unsuccess_perturbations_in_twenty)
    ax14.set(title="FAILURES: perturbation level = " + str(RANDOM_PERTURBATIONS_LEVEL) + " and intensity = "
                   + str(PERTURBATIONS_INTENSITY), ylabel="max number of perturbations in twenty consecutive steps")
    ax15.plot(range(unsuccessful_episodes_count), unsuccess_max_perturbations_in_twenty_run)
    ax15.set(ylabel="in which step")
    ax16.plot(range(unsuccessful_episodes_count), unsuccess_rewards)
    ax16.set(ylabel="Rewards")


def store_rewards(rewards, file_path):
    file_dump = open(file_path, "wb")
    pickle.dump(rewards, file_dump)


def show_fails_success_comparisson(RUNS, max_episode_steps, rewards, RANDOM_START_LEVEL, RANDOM_PERTURBATIONS_LEVEL,
                                   PERTURBATIONS_INTENSITY, INITIAL_POLE_ANGLE):
    if INITIAL_POLE_ANGLE == None:
        INITIAL_POLE_ANGLE = 0;

    rewards = np.asarray(rewards)

    fig, ax = plt.subplots()

    my_color = np.where(rewards == max_episode_steps, 'green', 'red')
    plt.scatter(range(RUNS), rewards, color=my_color, marker='x')
    ax.set(title="initial random level = " + str(RANDOM_START_LEVEL) + ', initial pole angle = ' + str(
        INITIAL_POLE_ANGLE) +
                 ', perturbation frequency = ' + str(RANDOM_PERTURBATIONS_LEVEL) + ', perturbation intensity std = ' +
                 str(PERTURBATIONS_INTENSITY), ylabel="cumulated reward", xlabel="episode")
    ax.plot(range(RUNS), rewards)
    plt.show()


def show_monitoring():
    plt.show()
