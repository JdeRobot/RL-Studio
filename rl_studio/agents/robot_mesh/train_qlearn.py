import datetime
import time

import gym

import numpy as np
from functools import reduce

from rl_studio.agents.f1.settings import QLearnConfig
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO

from rl_studio.algorithms.qlearn import QLearn
import rl_studio.agents.robot_mesh.utils as utils



class RobotMeshTrainer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # environment params
        self.params = params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        env_params = params.environment["params"]
        actions = params.environment["actions"]
        env_params["actions"] = actions
        self.env = gym.make(self.env_name, **env_params)
        # algorithm params
        self.alpha = params.algorithm["params"]["alpha"]
        self.epsilon = params.algorithm["params"]["epsilon"]
        self.gamma = params.algorithm["params"]["gamma"]
        self.config = QLearnConfig()
        self.stats = {}  # epoch: steps
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        self.outdir = "./logs/robot_mesh_experiments/"
        self.env = gym.wrappers.Monitor(self.env, self.outdir, force=True)
        self.actions = range(self.env.action_space.n)

        self.total_episodes = 20000
        self.epsilon_discount = 0.999  # Default 0.9986

        self.qlearn = QLearn(
            actions=self.actions, alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon
        )

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")

    def evaluate_and_learn_from_step(self, state):
        if self.qlearn.epsilon > 0.05:
            self.qlearn.epsilon *= self.epsilon_discount

        # Pick an action based on the current state
        action = self.qlearn.selectAction(state)

        print("Selected Action!! " + str(action))
        # Execute the action and get feedback
        nextState, reward, done, lap_completed = self.env.step(action)
        self.cumulated_reward += reward

        if  self.highest_reward <  self.cumulated_reward:
            self.highest_reward =  self.cumulated_reward

        # nextState = ''.join(map(str, observation))

        try:
            self.states_counter[nextState] += 1
        except KeyError:
            self.states_counter[nextState] = 1

        self.qlearn.learn(state, action, reward, nextState)

        self.env._flush(force=True)
        return nextState, done


    def main(self):

        self.print_init_info()

        if self.config.load_model:
            file_name = "1_20210701_0848_act_set_simple_epsilon_0.19_QTABLE.pkl"
            utils.load_model(self.params, self.qlearn, file_name)
            qvalues = np.array(list(self.qlearn.q.values()), dtype=np.float64)
            print(qvalues)
            self.highest_reward = max(qvalues)
        else:
            self.highest_reward = 0
        initial_epsilon = self.qlearn.epsilon

        telemetry_start_time = time.time()
        self.start_time = datetime.datetime.now()
        self.start_time_format = self.start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)
        for episode in range(self.total_episodes):

            self.cumulated_reward = 0
            state = self.env.reset()

            for step in range(50000):

                next_state, done = self.evaluate_and_learn_from_step(state);

                if not done:
                    state = next_state
                else:
                    last_time_steps = np.append(self.last_time_steps, [int(self.step + 1)])
                    self.stats[int(self.episode)] = self.step
                    self.states_reward[int(self.episode)] = self.cumulated_reward
                    print(
                        f"EP: {self.episode + 1} - epsilon: {round(self.qlearn.epsilon, 2)} - Reward: {self.cumulated_reward}"
                        f"- Time: {self.start_time_format} - Steps: {self.step}"
                    )
                    break

            if episode % 250 == 0 and self.config.save_model and episode > 1:
                print(f"\nSaving model . . .\n")
                utils.save_model(self.qlearn, self.start_time_format, self.stats, self.states_counter, self.states_reward)

            m, s = divmod(int(time.time() - telemetry_start_time), 60)
            h, m = divmod(m, 60)

        print(
            "Total EP: {} - epsilon: {} - ep. discount: {} - Highest Reward: {}".format(
                self.total_episodes, initial_epsilon, self.epsilon_discount, self.highest_reward
            )
        )

        l = last_time_steps.tolist()
        l.sort()

        # print("Parameters: a="+str)
        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
        print(
            "Best 100 score: {:0.2f}".format(
                reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])
            )
        )

        self.env.close()
