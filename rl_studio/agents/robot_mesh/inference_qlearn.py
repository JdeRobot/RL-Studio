import datetime
import multiprocessing
import time

import gym
import numpy as np

import rl_studio.agents.robot_mesh.utils as utils
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO
from rl_studio.wrappers.inference_rlstudio import InferencerWrapper


class QLearnRobotMeshInferencer:
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
        # algorithm param
        self.stats = {}  # epoch: steps
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)
        self.total_episodes = 20000

        self.outdir = "./logs/robot_mesh/"
        self.env = gym.wrappers.Monitor(self.env, self.outdir, force=True)

        inference_file = params.inference["params"]["inference_file"]
        actions_file = params.inference["params"]["actions_file"]
        self.highest_reward = 0

        self.inferencer = InferencerWrapper("qlearn", inference_file, actions_file)

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")

    def evaluate(self, state):
        # Pick an action based on the current state
        action = self.inferencer.inference(state)

        print("Selected Action!! " + str(action))
        # Execute the action and get feedback
        nextState, reward, done, lap_completed = self.env.step(action)
        self.cumulated_reward += reward

        if self.highest_reward < self.cumulated_reward:
            self.highest_reward = self.cumulated_reward

        try:
            self.states_counter[nextState] += 1
        except KeyError:
            self.states_counter[nextState] = 1

        self.env._flush(force=True)
        return nextState, done

    def simulation(self, queue):

        self.print_init_info()

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)
        for episode in range(self.total_episodes):

            self.cumulated_reward = 0
            state = self.env.reset()

            for step in range(50000):

                next_state, done = self.evaluate(state)

                if not done:
                    state = next_state
                else:
                    self.last_time_steps = np.append(
                        self.last_time_steps, [int(step + 1)]
                    )
                    self.stats[int(episode)] = step
                    self.states_reward[int(episode)] = self.cumulated_reward
                    print(
                        f"EP: {episode + 1}  - Reward: {self.cumulated_reward}"
                        f"- Time: {start_time_format} - Steps: {step}"
                    )
                    queue.put(step)
                    break

        print(
            "Total EP: {} - Highest Reward: {}".format(
                self.total_episodes, self.highest_reward
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
        figure, axes = utils.get_stats_figure(rewards)

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
                utils.update_line(axes, rewards)
