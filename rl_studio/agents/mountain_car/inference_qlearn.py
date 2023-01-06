import datetime
import multiprocessing
import time
from functools import reduce

import gym
import numpy as np

from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO
from rl_studio.wrappers.inference_rlstudio import InferencerWrapper
from . import utils as specific_utils


class QLearnMountainCarInferencer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # environment params
        self.n_steps = 0
        self.params = params
        self.environment_params = params["environments"]
        self.env_name = params["environments"]["env_name"]
        self.env = gym.make(self.env_name, **self.params)
        # algorithm params
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        inference_file = params["inference"]["inference_file"]
        actions_file = params["inference"]["actions_file"]

        self.highest_reward = 0
        self.total_episodes = 20000
        self.epsilon_discount = 0.999  # Default 0.9986
        self.cumulated_reward = 0

        self.inferencer = InferencerWrapper("qlearn", inference_file, actions_file)

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")

    def evaluate(self, state):
        action = self.inferencer.inference(state)

        print("Selected Action!! " + str(action))
        # Execute the action and get feedback
        if self.n_steps >= self.environment_params["max_steps"]:
            nextState, reward, done, info = self.env.step(-1)
        else:
            nextState, reward, done, info = self.env.step(action)
        self.n_steps = self.n_steps + 1
        print("step " + str(self.n_steps) + "!!!! ----------------------------")

        self.cumulated_reward += reward

        if self.highest_reward < self.cumulated_reward:
            self.highest_reward = self.cumulated_reward

        return nextState, done

    def simulation(self, queue):

        self.print_init_info()

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)

        for episode in range(self.total_episodes):

            self.n_steps = 0

            cumulated_reward = 0
            print("resetting")
            state = self.env.reset()

            for step in range(50000):

                next_state, done = self.evaluate(state)

                if not done:
                    state = next_state
                else:
                    self.last_time_steps = np.append(self.last_time_steps, [int(step + 1)])
                    self.states_reward[int(episode)] = cumulated_reward
                    print(
                        "---------------------------------------------------------------------------------------------"
                    )
                    print(
                        f"EP: {episode + 1} - Reward: {cumulated_reward}"
                        f"- Time: {start_time_format} - Steps: {step}"
                    )

                    queue.put(self.n_steps)

                    break

        print(
            "Total EP: {} - Highest Reward: {}".format(
                self.total_episodes, self.highest_reward
            )
        )

        l = self.last_time_steps.tolist()
        l.sort()

        print("Overall score: {:0.2f}".format(self.last_time_steps.mean()))
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
