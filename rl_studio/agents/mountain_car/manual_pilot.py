import datetime
import time

import sys as system
from envs.gazebo_envs import *

import numpy as np
from functools import reduce

from agents.f1 import settings as settings

# from gym.envs.registration import register
import utils

# # my envs
# register(
#     id='mySim-v0',
#     entry_point='envs:MyEnv',
#     # More arguments here
# )

if __name__ == "__main__":

    print(settings.title)
    print(settings.description)
    print(f"\t- Start hour: {datetime.datetime.now()}")

    environment = settings.envs_params["simple"]
    print(environment)
    env = gym.make(environment["env"], **environment)

    # TODO: Move to settings file
    outdir = "./logs/robot_mesh_experiments/"
    stats = {}  # epoch: steps
    states_counter = {}
    states_reward = {}

    # plotter = liveplot.LivePlot(outdir)

    last_time_steps = np.ndarray(0)

    actions = range(env.action_space.n)
    env = gym.wrappers.Monitor(env, outdir, force=True)
    counter = 0
    estimate_step_per_lap = environment["estimated_steps"]
    lap_completed = False
    total_episodes = 20000
    epsilon_discount = 0.999  # Default 0.9986
    rewards_per_run = [0, 0]
    env.done = False

    figure, axes = utils.get_stats_figure(rewards_per_run)

    # START ############################################################################################################
    for episode in range(total_episodes):

        counter = 0
        lap_completed = False

        initial_epsilon = 0.999
        highest_reward = 0
        cumulated_reward = 0
        print("resetting")
        time.sleep(5)
        state = env.reset()

        # state = ''.join(map(str, observation))

        for step in range(50000):

            inpt = input("provide action (0-none, 1-left, 2-right): ")
            if inpt == "1" or inpt == "2" or inpt == "0":
                action = int(inpt)
                print("Selected Action!! " + str(action))
                # Execute the action and get feedback
                nextState, reward, env.done, lap_completed = env.step(action)
                cumulated_reward += reward

                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward
                print("cumulated_reward = " + str(cumulated_reward))
                env._flush(force=True)
                if not env.done:
                    state = nextState
                else:
                    break
            elif inpt == "q":
                system.exit(1)
            else:
                nextState, reward, env.done, lap_completed = env.step(-1)
                print("wrong action! Try again")
                break

            rewards_per_run.append(cumulated_reward)
            axes.cla()
            utils.update_line(axes, rewards_per_run)

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
