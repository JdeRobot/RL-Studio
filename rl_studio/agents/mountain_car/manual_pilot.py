import datetime
import sys as system
import time

import numpy as np

import utils
from rl_studio.agents.f1 import settings as settings


def execute_steps():
    highest_reward = 0
    cumulated_reward = 0
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
            if env.done:
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

    for episode in range(total_episodes):
        counter = 0
        lap_completed = False

        print("resetting")
        time.sleep(5)
        state = env.reset()

        execute_steps()

    env.close()
