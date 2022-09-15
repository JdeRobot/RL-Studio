import datetime
import time
import random

import gym
from rl_studio.wrappers.inference_rlstudio import InferencerWrapper
from tqdm import tqdm

from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO


class DQNCartpoleInferencer:
    def __init__(self, params):

        self.now = datetime.datetime.now()
        # self.environment params
        self.params = params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        self.config = params.settings["params"]
        random_start_level = self.environment_params["random_start_level"]

        self.env = gym.make(self.env_name, random_start_level=random_start_level)
        self.RUNS = self.environment_params["runs"]
        self.UPDATE_EVERY = self.environment_params[
            "update_every"
        ]  # How oftern the current progress is recorded
        self.RANDOM_PERTURBATIONS_LEVEL = self.environment_params.get("random_perturbations_level", 0)

        self.actions = self.env.action_space.n

        self.losses_list, self.reward_list, self.episode_len_list, self.epsilon_list = (
            [],
            [],
            [],
            [],
        )  # metrics recorded for graph
        self.epsilon = 0

        inference_file = params.inference["params"]["inference_file"]
        # TODO the first parameter (algorithm) should come from configuration
        self.inferencer = InferencerWrapper("dqn", inference_file, env=self.env)

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- self.environment params:\n{self.environment_params}")

    def main(self):

        self.print_init_info()

        epoch_start_time = datetime.datetime.now()

        print(LETS_GO)
        total_reward_in_epoch = 0
        for run in tqdm(range(self.RUNS)):
            obs, done, rew = self.env.reset(), False, 0
            while not done:
                A = self.inferencer.inference(obs)
                obs, reward, done, info = self.env.step(A.item())
                rew += reward
                total_reward_in_epoch += reward
                time.sleep(0.01)
                if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                    self.env.step(random.randrange(self.env.action_space.n))
                self.env.render()

            # monitor progress
            if run % self.UPDATE_EVERY == 0:
                time_spent = datetime.datetime.now() - epoch_start_time
                epoch_start_time = datetime.datetime.now()
                print(
                    "\nRun:",
                    run,
                    "Average:",
                    total_reward_in_epoch / self.UPDATE_EVERY,
                    "time spent",
                    time_spent,
                )
                total_reward_in_epoch = 0
