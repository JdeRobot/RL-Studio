import datetime
import time

import gym

import numpy as np
from functools import reduce

# import liveplot
import multiprocessing


from rl_studio.agents.f1.settings import QLearnConfig
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO

from rl_studio.algorithms.qlearn_two_states import QLearn
from . import utils as specific_utils
import rl_studio.agents.mountain_car.utils as utils


class MountainCarTrainer:
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
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        self.config = QLearnConfig()
        self.outdir = "./logs/robot_mesh_experiments/"
        self.env = gym.wrappers.Monitor(self.env, self.outdir, force=True)
        self.actions = range(self.env.action_space.n)
        self.env.done = True

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
        if self.qlearn.epsilon > 0.01:
            self.qlearn.epsilon *= self.epsilon_discount
            print("epsilon = " + str(self.qlearn.epsilon))

        # Pick an action based on the current state
        action = self.qlearn.selectAction(state)

        print("Selected Action!! " + str(action))
        # Execute the action and get feedback
        if self.n_steps >= self.environment_params["max_steps"]:
            nextState, reward, done, info = self.env.step(-1)
        else:
            nextState, reward, done, info = self.env.step(action)
        n_steps = self.n_steps + 1
        print("step " + str(n_steps) + "!!!! ----------------------------")

        self.cumulated_reward += reward

        if self.highest_reward < self.cumulated_reward:
            self.highest_reward = self.cumulated_reward

        self.qlearn.learn(state, action, reward, nextState, done)

        self.env._flush(force=True)
        return nextState, done

    def simulation(self, queue):

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
        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)

        for episode in range(self.total_episodes):

            done = False
            self.n_steps = 0

            cumulated_reward = 0
            print("resetting")
            state = self.env.reset()

            # state = ''.join(map(str, observation))

            for step in range(50000):


                next_state, done = self.evaluate_and_learn_from_step(state);

                if not done:
                    state = next_state
                else:
                    last_time_steps = np.append(last_time_steps, [int(step + 1)])
                    self.stats[int(episode)] = step
                    self.states_reward[int(episode)] = cumulated_reward
                    print(
                        "---------------------------------------------------------------------------------------------"
                    )
                    print(
                        f"EP: {episode + 1} - epsilon: {round(self.qlearn.epsilon, 2)} - Reward: {cumulated_reward}"
                        f"- Time: {start_time_format} - Steps: {step}"
                    )

                    # get_stats_figure(rewards_per_run)
                    self.rewards_per_run.append(cumulated_reward)
                    queue.put(self.n_steps)

                    break

                if episode % 250 == 0 and self.config.save_model and episode > 1:
                    print(f"\nSaving model . . .\n")
                    utils.save_model(self.qlearn, start_time_format, self.stats, self.states_counter, self.states_reward)



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

    def main(self):
        # Create a queue to share data between process
        queue = multiprocessing.Queue()

        # Create and start the simulation process
        simulate = multiprocessing.Process(None, self.simulation, args=(queue,))
        simulate.start()

        rewards = []
        while queue.empty():
            time.sleep(5)
        # Call a function to update the plot when there is new data
        result = queue.get(block=True, timeout=None)
        rewards.append(result)
        figure, axes = specific_utils.get_stats_figure(rewards)

        while True:
            while queue.empty():
                time.sleep(5)
            # Call a function to update the plot when there is new data
            result = queue.get(block=True, timeout=None)
            if result != None:
                print(
                    "PLOT: Received reward to paint!!! -> REWARD PAINTED = "
                    + str(result)
                )
                rewards.append(result)
                axes.cla()
                specific_utils.update_line(axes, rewards)
