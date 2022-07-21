import datetime
import pickle

from rl_studio.agents.f1 import settings


#TODO Since these utils are algorithm specific, those should stay in the algorithm folder somehow tied to its algorithm class


def load_model(params, qlearn, file_name):

    qlearn_file = open("./logs/qlearn_models/" + file_name)
    model = pickle.load(qlearn_file)

    qlearn.q = model
    qlearn.alpha = params.algorithm["params"]["alpha"]
    qlearn.gamma = params.algorithm["params"]["epsilon"]
    qlearn.epsilon = params.algorithm["params"]["gamma"]

    # highest_reward = settings.algorithm_params["highest_reward"]

    print(f"\n\nMODEL LOADED. Number of (action, state): {len(model)}")
    print(f"    - Loading:    {file_name}")
    print(f"    - Model size: {len(qlearn.q)}")
    print(f"    - Action set: {settings.actions_set}")
    print(f"    - Epsilon:    {qlearn.epsilon}")
    print(f"    - Start:      {datetime.datetime.now()}")


def save_model(qlearn, current_time, states, states_counter, states_rewards):
    # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
    # (S x A), where s are all states, a are all the possible actions. After the environment is solved, just save this
    # matrix as a csv file. I have a quick implementation of this on my GitHub under Reinforcement Learning.

    # Q TABLE
    base_file_name = "_act_set_{}_epsilon_{}".format(
        settings.qlearn.actions_set, round(qlearn.epsilon, 2)
    )
    file_dump = open(f"./logs/qlearn_models/1_{current_time}{base_file_name}_QTABLE.pkl", "wb")
    pickle.dump(qlearn.q, file_dump)
    # STATES COUNTER
    states_counter_file_name = base_file_name + "_STATES_COUNTER.pkl"
    file_dump = open(f"./logs/qlearn_models/2_{current_time}{states_counter_file_name}", "wb")
    pickle.dump(states_counter, file_dump)
    # STATES CUMULATED REWARD
    states_cum_reward_file_name = base_file_name + "_STATES_CUM_REWARD.pkl"
    file_dump = open(f"./logs/qlearn_models/3_{current_time}{states_cum_reward_file_name}", "wb")
    pickle.dump(states_rewards, file_dump)
    # STATES
    steps = base_file_name + "_STATES_STEPS.pkl"
    file_dump = open(f"./logs/qlearn_models/4_{current_time}{steps}", "wb")
    pickle.dump(states, file_dump)


def save_times(checkpoints):
    file_name = "actions_"
    file_dump = open(f"/logs/{file_name}{settings.qlearn.actions_set}_checkpoints.pkl", "wb")
    pickle.dump(checkpoints, file_dump)

def save_actions(actions, current_time):
    file_dump = open(
        "./logs/qlearn_models/actions_set_" + current_time, "wb"
    )
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
