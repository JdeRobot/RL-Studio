import datetime
import time
import random

import gym

import logging

from rl_studio.agents.cartpole.utils import plot_random_perturbations_monitoring, plot_random_start_level_monitoring, \
    show_monitoring, plot_fails_success_comparisson
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
        self.RANDOM_START_LEVEL = self.environment_params["random_start_level"]

        if self.config["logging_level"] == "debug":
            self.LOGGING_LEVEL = logging.DEBUG
        elif self.config["logging_level"] == "error":
            self.LOGGING_LEVEL = logging.ERROR
        elif self.config["logging_level"] == "critical":
            self.LOGGING_LEVEL = logging.CRITICAL
        else:
            self.LOGGING_LEVEL = logging.INFO

        self.env = gym.make(self.env_name, random_start_level=self.RANDOM_START_LEVEL)
        self.RUNS = self.environment_params["runs"]
        self.SHOW_EVERY = self.environment_params[
            "show_every"
        ]
        self.UPDATE_EVERY = self.environment_params[
            "update_every"
        ]  # How oftern the current progress is recorded
        self.RANDOM_PERTURBATIONS_LEVEL = self.environment_params.get("random_perturbations_level", 0)
        self.PERTURBATIONS_INTENSITY = self.environment_params.get("perturbations_intensity", 0)

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
        logging.info(JDEROBOT)
        logging.info(JDEROBOT_LOGO)
        logging.info(f"\t- Start hour: {datetime.datetime.now()}\n")
        logging.info(f"\t- self.environment params:\n{self.environment_params}")

    def main(self):
        epoch_start_time = datetime.datetime.now()

        logs_dir = 'logs/cartpole/dqn/inference/'
        logs_file_name = 'logs_file_' + str(self.RANDOM_START_LEVEL) + '_' + str(
            self.RANDOM_PERTURBATIONS_LEVEL) + '_' + str(epoch_start_time) \
                         + str(self.PERTURBATIONS_INTENSITY) + '.log'
        logging.basicConfig(filename=logs_dir + logs_file_name, filemode='a',
                            level=self.LOGGING_LEVEL,
                            format='%(name)s - %(levelname)s - %(message)s')
        self.print_init_info()

        unsuccessful_episodes_count = 0
        unsuccessful_initial_states = []
        successful_initial_states = []
        success_rewards = []
        unsuccess_rewards = []
        success_perturbations_in_twenty = []
        unsuccess_perturbations_in_twenty = []
        success_max_perturbations_in_twenty_run = []
        unsuccess_max_perturbations_in_twenty_run = []
        last_ten_steps = []

        logging.info(LETS_GO)
        total_reward_in_epoch = 0
        for run in tqdm(range(self.RUNS)):
            max_perturbations_in_twenty = 0
            max_perturbations_run = 0
            last_perturbation_action = 0

            obs, done, rew = self.env.reset(), False, 0
            initial_state = obs
            while not done:
                A = self.inferencer.inference(obs)
                obs, reward, done, info = self.env.step(A.item())
                rew += reward
                total_reward_in_epoch += reward
                time.sleep(0.01)

                if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                    perturbation_action = random.randrange(self.env.action_space.n)
                    for perturbation in range(self.PERTURBATIONS_INTENSITY):
                        self.env.perturbate(perturbation_action)
                    logging.info("perturbated in step {} with action {}".format(rew, perturbation_action))

                    if rew > 20:
                        last_ten_steps.append(1)
                        if len(last_ten_steps) > 20:
                            last_ten_steps.pop(0)
                else:
                    if rew > 20:
                        last_ten_steps.append(0)
                        if len(last_ten_steps) > 20:
                            last_ten_steps.pop(0)

                if max_perturbations_in_twenty < sum(last_ten_steps):
                    max_perturbations_in_twenty = sum(last_ten_steps)
                    max_perturbations_run = rew

                if run % self.SHOW_EVERY == 0:
                    self.env.render()

            # monitor progress
            if (run+1) % self.UPDATE_EVERY == 0:
                time_spent = datetime.datetime.now() - epoch_start_time
                epoch_start_time = datetime.datetime.now()
                updates_message = 'Run: {0} Average: {1} time spent {2}'.format(run,
                                                                                total_reward_in_epoch / self.UPDATE_EVERY,
                                                                                str(time_spent))
                logging.info(updates_message)
                print(updates_message)
                total_reward_in_epoch = 0

            if rew < 500:
                unsuccessful_episodes_count += 1

                unsuccessful_initial_states.append(initial_state)
                unsuccess_rewards.append(rew)
                unsuccess_perturbations_in_twenty.append(max_perturbations_in_twenty)
                unsuccess_max_perturbations_in_twenty_run.append(max_perturbations_run)
            else:
                successful_initial_states.append(initial_state)
                success_rewards.append(rew)
                success_perturbations_in_twenty.append(max_perturbations_in_twenty)
                success_max_perturbations_in_twenty_run.append(max_perturbations_run)

        if self.RANDOM_START_LEVEL > 0:
            plot_random_start_level_monitoring(unsuccessful_episodes_count, unsuccessful_initial_states,
                                               unsuccess_rewards, success_rewards, successful_initial_states,
                                               self.RUNS, self.RANDOM_START_LEVEL)

        if self.RANDOM_PERTURBATIONS_LEVEL > 0:
            plot_random_perturbations_monitoring(unsuccessful_episodes_count, success_perturbations_in_twenty,
                                                 success_max_perturbations_in_twenty_run, success_rewards,
                                                 unsuccess_perturbations_in_twenty,
                                                 unsuccess_max_perturbations_in_twenty_run, unsuccess_rewards,
                                                 self.RUNS, self.RANDOM_PERTURBATIONS_LEVEL,
                                                 self.PERTURBATIONS_INTENSITY)

        plot_fails_success_comparisson(unsuccessful_episodes_count, success_rewards, unsuccess_rewards,
                                                 self.RUNS, self.RANDOM_START_LEVEL, self.RANDOM_PERTURBATIONS_LEVEL,
                                                 self.PERTURBATIONS_INTENSITY);

        show_monitoring()
