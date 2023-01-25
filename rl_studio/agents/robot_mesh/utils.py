import pickle

import matplotlib.pyplot as plt
import pandas as pd


# TODO Since these utils are algorithm specific, those should stay in the algorithm folder somehow tied to its algorithm class


# TODO Since these utils are algorithm specific, those should stay in the algorithm folder somehow tied to its algorithm class


def update_line(axes, runs_rewards):
    plot_rewards_per_run(axes, runs_rewards)
    plt.draw()
    plt.pause(0.01)


def get_stats_figure(runs_rewards):
    fig, axes = plt.subplots()
    fig.set_size_inches(12, 4)
    plot_rewards_per_run(axes, runs_rewards)
    plt.ion()
    plt.show()
    return fig, axes


def plot_rewards_per_run(axes, runs_rewards):
    rewards_graph = pd.DataFrame(runs_rewards)
    ax = rewards_graph.plot(ax=axes, title="steps per run")
    ax.set_xlabel("runs")
    ax.set_ylabel("steps")
    ax.legend().set_visible(False)


def save_model(qlearn, current_time, states, states_counter, states_rewards):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.

    # TODO The paths are not relative to the agents folder
    # Q TABLE
    base_file_name = "_epsilon_{}".format(round(qlearn.epsilon, 3))
    file_dump = open(
        "./logs/robot_mesh/1_" + current_time + base_file_name + "_QTABLE.pkl", "wb"
    )
    pickle.dump(qlearn.q, file_dump)
    # STATES COUNTER
    states_counter_file_name = base_file_name + "_STATES_COUNTER.pkl"
    file_dump = open(
        "./logs/robot_mesh/2_" + current_time + states_counter_file_name, "wb"
    )
    pickle.dump(states_counter, file_dump)
    # STATES CUMULATED REWARD
    states_cum_reward_file_name = base_file_name + "_STATES_CUM_REWARD.pkl"
    file_dump = open(
        "./logs/robot_mesh/3_" + current_time + states_cum_reward_file_name, "wb"
    )
    pickle.dump(states_rewards, file_dump)
    # STATES
    steps = base_file_name + "_STATES_STEPS.pkl"
    file_dump = open("./logs/robot_mesh/4_" + current_time + steps, "wb")
    pickle.dump(states, file_dump)


def save_actions(actions, start_time):
    file_dump = open("./logs/robot_mesh/actions_set_" + start_time, "wb")
    pickle.dump(actions, file_dump)


def render(env, episode):
    render_skip = 0
    render_interval = 50
    render_episodes = 10

    if (episode % render_interval == 0) and (episode != 0) and (episode > render_skip):
        env.render()
    elif (
        ((episode - render_episodes) % render_interval == 0)
        and (episode != 0)
        and (episode > render_skip)
        and (render_episodes < episode)
    ):
        env.render(close=True)
