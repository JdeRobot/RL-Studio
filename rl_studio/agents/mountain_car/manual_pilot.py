import datetime
import sys as system
import time

import gym

import utils
from rl_studio.agents.f1 import settings as settings


class ManualMountainCarTrainer:
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
        self.rewards_per_run = [0, 0]

    def execute_steps(self):
        highest_reward = 0
        cumulated_reward = 0
        for step in range(50000):

            inpt = input("provide action (0-none, 1-left, 2-right): ")
            if inpt == "1" or inpt == "2" or inpt == "0":
                action = int(inpt)
                print("Selected Action!! " + str(action))
                # Execute the action and get feedback
                nextState, reward, self.envdone, lap_completed = self.envstep(action)
                cumulated_reward += reward

                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward
                print("cumulated_reward = " + str(cumulated_reward))
                self.env_flush(force=True)
                if self.envdone:
                    break
            elif inpt == "q":
                system.exit(1)
            else:
                nextState, reward, self.envdone, lap_completed = self.envstep(-1)
                print("wrong action! Try again")
                break

            self.rewards_per_run.append(cumulated_reward)

    def main(self):

        print(settings.title)
        print(settings.description)
        print(f"\t- Start hour: {datetime.datetime.now()}")

        total_episodes = 20000
        self.env.done = False

        utils.get_stats_figure(self.rewards_per_run)

        for episode in range(total_episodes):
            print("resetting")
            time.sleep(5)
            self.env.reset()

            self.execute_steps()

        self.env.close()
