import datetime
import time

import sys as system
from envs.gazebo_envs import *

import numpy as np
from functools import reduce

from agents.f1.settings import QLearnConfig

# from gym.envs.registration import register

# # my envs
# register(
#     id='mySim-v0',
#     entry_point='envs:MyEnv',
#     # More arguments here
# )


class RobotMeshTrainer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # environment params
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
        # agent
        # self.action_number = params.agent["params"]["actions_number"]
        # self.actions_set = params.agent["params"]["actions_set"]
        # self.actions_values = params.agent["params"]["available_actions"][self.actions_set]

    def main(self):

        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")
        config = QLearnConfig()

        # TODO: Move to settings file
        outdir = "./logs/robot_mesh_experiments/"
        stats = {}  # epoch: steps
        states_counter = {}
        states_reward = {}

        last_time_steps = np.ndarray(0)

        env = gym.wrappers.Monitor(self.env, outdir, force=True)
        actions = range(env.action_space.n)

        print(f"\t- Start hour: {datetime.datetime.now()}")

        # TODO: Move to settings file
        outdir = "./logs/robot_mesh_experiments/"
        stats = {}  # epoch: steps
        states_counter = {}
        states_reward = {}

        # plotter = liveplot.LivePlot(outdir)

        last_time_steps = np.ndarray(0)

        actions = range(env.action_space.n)
        env = gym.wrappers.Monitor(self.env, outdir, force=True)
        counter = 0
        lap_completed = False
        total_episodes = 20000
        epsilon_discount = 0.999  # Default 0.9986
        rewards_per_run = [0, 0]
        env.done = False

        # START ############################################################################################################
        for episode in range(total_episodes):

            counter = 0
            lap_completed = False

            initial_epsilon = 0.999
            highest_reward = 0
            cumoviulated_reward = 0
            print("resetting")
            time.sleep(5)
            state = env.reset()

            # state = ''.join(map(str, observation))

            for step in range(50000):

                inpt = input("provide action (0-up, 1-right, 2-down, 3-left): ")
                if inpt == "0" or inpt == "1" or inpt == "2" or inpt == "3":
                    action = int(inpt)
                    print("Selected Action!! " + str(action))
                    # Execute the action and get feedback
                    nextState, reward, env.done, lap_completed = env.step(action)

                    env._flush(force=True)
                    if not env.done:
                        state = nextState
                    else:
                        break
                elif inpt == "q":
                    system.exit(1)
                else:
                    print("wrong action! Try again")

        print(
            "Total EP: {} - epsilon: {} - ep. discount: {} - Highest Reward: {}".format(
                total_episodes, initial_epsilon, epsilon_discount, highest_reward
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

        env.close()
