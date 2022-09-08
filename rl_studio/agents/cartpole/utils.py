import pickle

import numpy as np

# How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded
LEARNING_RATE = 0.1
# Between 0 and 1, mesue of how much we carre about future reward over immedate reward
DISCOUNT = 0.95
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 10000 // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def save_model(qlearn, current_time, states, states_counter, states_rewards):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.

    # TODO The paths are not relative to the agents folder
    # Q TABLE
    base_file_name = "_epsilon_{}".format(round(qlearn.epsilon, 3))
    file_dump = open(
        "./logs/qlearn_models/1_" + current_time + base_file_name + "_QTABLE.pkl", "wb"
    )
    pickle.dump(qlearn.q, file_dump)
    # STATES COUNTER
    states_counter_file_name = base_file_name + "_STATES_COUNTER.pkl"
    file_dump = open(
        "./logs/qlearn_models/2_" + current_time + states_counter_file_name, "wb"
    )
    pickle.dump(states_counter, file_dump)
    # STATES CUMULATED REWARD
    states_cum_reward_file_name = base_file_name + "_STATES_CUM_REWARD.pkl"
    file_dump = open(
        "./logs/qlearn_models/3_" + current_time + states_cum_reward_file_name, "wb"
    )
    pickle.dump(states_rewards, file_dump)
    # STATES
    steps = base_file_name + "_STATES_STEPS.pkl"
    file_dump = open("./logs/qlearn_models/4_" + current_time + steps, "wb")
    pickle.dump(states, file_dump)


def save_dqn_model(dqn, current_time, average):
    base_file_name = "_epsilon_{}".format(round(epsilon, 2))
    file_dump = open(
        "./logs/dqn_models/1_" + current_time + base_file_name + "_DQN_WEIGHTS_avg_" + str(average) + ".pkl",
        "wb",
    )
    pickle.dump(dqn.q_net, file_dump)


def save_actions(actions, start_time):
    file_dump = open("./logs/qlearn_models/actions_set_" + start_time, "wb")
    pickle.dump(actions, file_dump)


# Create bins and Q table
def create_bins_and_q_table(env):
    # env.observation_space.high
    # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
    # env.observation_space.low
    # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

    # remove hard coded Values when I know how to

    numBins = 20
    obsSpaceSize = len(env.observation_space.high)

    # Get the size of each bucket
    bins = [
        np.linspace(-4.8, 4.8, numBins),
        np.linspace(-4, 4, numBins),
        np.linspace(-0.418, 0.418, numBins),
        np.linspace(-4, 4, numBins),
    ]

    qTable = np.random.uniform(
        low=-2, high=0, size=([numBins] * obsSpaceSize + [env.action_space.n])
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
