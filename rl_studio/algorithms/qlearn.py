import os
import pickle
import random
import time

import numpy as np


class QLearnF1:
    def __init__(
        self, states_len, actions, actions_len, epsilon, alpha, gamma, num_regions
    ):
        self.q_table = np.random.uniform(
            low=0, high=0, size=([num_regions + 1] * states_len + [actions_len])
        )
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.actions = actions
        self.actions_len = actions_len

    def select_action(self, state):
        state = state[0]

        if np.random.random() > self.epsilon:
            # Get action from Q table
            action = np.argmax(self.q_table[state])
        else:
            # Get random action
            action = np.random.randint(0, self.actions_len)

        return action

    def learn(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = next_state[0]

        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state + (action,)]
        new_q = (1 - self.alpha) * current_q + self.alpha * (
            reward + self.gamma * max_future_q
        )

        # Update Q table with new Q value
        self.q_table[state + (action,)] = new_q

    def inference(self, state):
        return np.argmax(self.q_table[state])

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon
        return self.epsilon

    def load_table(self, file):
        self.q_table = np.load(file)

    def save_numpytable(
        self,
        qtable,
        environment,
        outdir,
        cumulated_reward,
        episode,
        step,
        epsilon,
    ):
        # Q Table as Numpy
        # np_file = (
        #    f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{environment['circuit_name']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}-qtable.npy",
        # )
        # qtable = np.array([list(item.values()) for item in self.q.values()])
        np.save(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{environment['circuit_name']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}-qtable.npy",
            qtable,
        )

    def save_model(
        self,
        environment,
        outdir,
        qlearn,
        cumulated_reward,
        episode,
        step,
        epsilon,
        states,
        states_counter,
        states_rewards,
    ):
        # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
        # (S x A), where s are all states, a are all the possible actions.

        # outdir_models = f"{outdir}_models"
        os.makedirs(f"{outdir}", exist_ok=True)

        # Q TABLE PICKLE
        # base_file_name = "_actions_set:_{}_epsilon:_{}".format(settings.actions_set, round(qlearn.epsilon, 2))
        base_file_name = f"_Circuit-{environment['circuit_name']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}"
        file_dump = open(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}_QTABLE.pkl",
            "wb",
        )
        pickle.dump(qlearn.q, file_dump)

        # STATES COUNTER
        # states_counter_file_name = base_file_name + "_STATES_COUNTER.pkl"
        file_dump = open(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}_STATES_COUNTER.pkl",
            "wb",
        )
        pickle.dump(states_counter, file_dump)

        # STATES CUMULATED REWARD
        # states_cum_reward_file_name = base_file_name + "_STATES_CUM_REWARD.pkl"
        file_dump = open(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}_STATES_CUM_REWARD.pkl",
            "wb",
        )
        pickle.dump(states_rewards, file_dump)

        # STATES
        # steps = base_file_name + "_STATES_STEPS.pkl"
        file_dump: BufferedWriter = open(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}_STATES_CUM_REWARD.pkl",
            "wb",
        )
        pickle.dump(states, file_dump)

        # Q Table as Numpy
        # np_file = (
        #    f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}-qtable.npy",
        # )
        # qtable = np.array([list(item.values()) for item in self.q.values()])
        # np.save(np_file, qtable)


class QLearn:
    def __init__(self, actions, epsilon=0.99, alpha=0.8, gamma=0.9):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # discount constant
        self.gamma = gamma  # discount factor
        self.actions = actions

    def getQValues(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        """
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        """
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def selectAction(self, state, return_q=False):
        q = [self.getQValues(state, a) for a in self.actions]
        maxQ = max(q)
        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [
                q[i] + random.random() * mag - 0.5 * mag
                for i in range(len(self.actions))
            ]
            maxQ = max(q)
        count = q.count(maxQ)

        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = i

        if return_q:  # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQValues(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def inference(self, state, return_q=False):
        q = [self.getQValues(state, a) for a in self.actions]
        maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:  # if they want it, give it!
            return action, q
        return action

    def load_pickle_model(self, file_path):

        qlearn_file = open(file_path, "rb")
        self.q = pickle.load(qlearn_file)

    def load_np_model(self, file):
        self.q = np.load(file)

    def save_model(
        self,
        environment,
        outdir,
        qlearn,
        cumulated_reward,
        episode,
        step,
        epsilon,
        states,
        states_counter,
        states_rewards,
    ):
        # Tabular RL: Tabular Q-learning basically stores the policy (Q-values) of  the agent into a matrix of shape
        # (S x A), where s are all states, a are all the possible actions.

        # outdir_models = f"{outdir}_models"
        os.makedirs(f"{outdir}", exist_ok=True)

        # Q TABLE PICKLE
        # base_file_name = "_actions_set:_{}_epsilon:_{}".format(settings.actions_set, round(qlearn.epsilon, 2))
        base_file_name = f"_Circuit-{environment['circuit_name']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}"
        file_dump = open(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}_QTABLE.pkl",
            "wb",
        )
        pickle.dump(qlearn.q, file_dump)

        # STATES COUNTER
        # states_counter_file_name = base_file_name + "_STATES_COUNTER.pkl"
        file_dump = open(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}_STATES_COUNTER.pkl",
            "wb",
        )
        pickle.dump(states_counter, file_dump)

        # STATES CUMULATED REWARD
        # states_cum_reward_file_name = base_file_name + "_STATES_CUM_REWARD.pkl"
        file_dump = open(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}_STATES_CUM_REWARD.pkl",
            "wb",
        )
        pickle.dump(states_rewards, file_dump)

        # STATES
        # steps = base_file_name + "_STATES_STEPS.pkl"
        file_dump: BufferedWriter = open(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}_STATES_CUM_REWARD.pkl",
            "wb",
        )
        pickle.dump(states, file_dump)

        # Q Table as Numpy
        # np_file = (
        #    f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_{base_file_name}-qtable.npy",
        # )
        # qtable = np.array([list(item.values()) for item in self.q.values()])
        # np.save(np_file, qtable)

    def load_qmodel_actionsmodel(self, file_path, actions_path):

        qlearn_file = open(file_path, "rb")
        actions_file = open(actions_path, "rb")

        self.q = pickle.load(qlearn_file)
        # TODO it may be possible to infer the actions from the model. I don know enough to assume that for every algorithm
        self.actions = pickle.load(actions_file)

        print(f"\n\nMODEL LOADED.")
        print(f"    - Loading:    {file_path}")
        print(f"    - Model size: {len(self.q)}")

    def updateEpsilon(self, epsilon):
        self.epsilon = epsilon
        return self.epsilon


class QLearnCarla:
    def __init__(
        self, states_len, actions, actions_len, epsilon, alpha, gamma, num_regions
    ):
        self.q_table = np.random.uniform(
            low=0, high=0, size=([num_regions + 1] * states_len + [actions_len])
        )
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.actions = actions
        self.actions_len = actions_len

    def select_action(self, state):

        # print(f"in selec_action()")
        # print(f"qlearn.q_table = {self.q_table}")
        # print(f"len qlearn.q_table = {len(self.q_table)}")
        # print(f"type qlearn.q_table = {type(self.q_table)}")
        # print(f"shape qlearn.q_table = {np.shape(self.q_table)}")
        # print(f"size qlearn.q_table = {np.size(self.q_table)}")

        ## ONLY TAKES 1 STATE
        state = state[0]

        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.randint(0, self.actions_len)

        return action

    def learn(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = next_state[0]

        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state + (action,)]
        new_q = (1 - self.alpha) * current_q + self.alpha * (
            reward + self.gamma * max_future_q
        )

        # Update Q table with new Q value
        self.q_table[state + (action,)] = new_q

    def inference(self, state):
        return np.argmax(self.q_table[state])

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon
        return self.epsilon

    def load_table(self, file):
        self.q_table = np.load(file)

    def save_numpytable(
        self,
        qtable,
        environment,
        outdir,
        cumulated_reward,
        episode,
        step,
        epsilon,
    ):
        np.save(
            f"{outdir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{environment['circuit_name']}_States-{environment['states']}_Actions-{environment['action_space']}_Rewards-{environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}-qtable.npy",
            qtable,
        )
