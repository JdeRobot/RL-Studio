import datetime
import time
import random

import gym

import logging
import numpy as np

from rl_studio.agents.cartpole.cartpole_Inferencer import CartpoleInferencer
from rl_studio.agents.cartpole.utils import store_rewards, show_fails_success_comparisson
from rl_studio.wrappers.inference_rlstudio import InferencerWrapper
from tqdm import tqdm

from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO


class DQNCartpoleInferencer(CartpoleInferencer):
    def __init__(self, params):
        super().__init__(params);
        self.OBJECTIVE = self.environment_params[
            "objective_reward"
        ]

        self.actions = self.env.action_space.n

        self.losses_list, self.reward_list, self.episode_len_list, self.epsilon_list = (
            [],
            [],
            [],
            [],
        )  # metrics recorded for graph
        self.epsilon = 0

        inference_file = params["inference"]["inference_file"]
        # TODO the first parameter (algorithm) should come from configuration
        self.inferencer = InferencerWrapper("dqn", inference_file, env=self.env)

    def print_init_info(self):
        logging.info(JDEROBOT)
        logging.info(JDEROBOT_LOGO)
        logging.info(f"\t- Start hour: {datetime.datetime.now()}\n")
        logging.info(f"\t- self.environment params:\n{self.environment_params}")

    def run_experiment(self):
        epoch_start_time = datetime.datetime.now()

        logs_dir = 'logs/cartpole/dqn/inference/'
        logs_file_name = 'logs_file_' + str(self.RANDOM_START_LEVEL) + '_' + str(
            self.RANDOM_PERTURBATIONS_LEVEL) + '_' + str(epoch_start_time) \
                         + str(self.PERTURBATIONS_INTENSITY_STD) + '.log'
        logging.basicConfig(filename=logs_dir + logs_file_name, filemode='a',
                            level=self.LOGGING_LEVEL,
                            format='%(name)s - %(levelname)s - %(message)s')
        self.print_init_info()

        unsuccessful_episodes_count = 0
        episodes_rewards = []

        logging.info(LETS_GO)
        total_reward_in_epoch = 0
        total_secs=0
        for run in tqdm(range(self.RUNS)):
            obs, done, rew = self.env.reset(), False, 0
            while not done:
                if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                    perturbation_action = random.randrange(self.env.action_space.n)
                    obs, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                    logging.info("perturbated in step {} with action {}".format(rew, perturbation_action))
                if self.RANDOM_PERTURBATIONS_LEVEL > 1 and  random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL - 1:
                    perturbation_action = random.randrange(2)
                    state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                    logging.debug("perturbated in step {} with action {}".format(rew, perturbation_action))


                A = self.inferencer.inference(obs)
                obs, reward, done, info = self.env.step(A.item())
                total_secs+=info["time"]

                rew += reward
                total_reward_in_epoch += reward

                if run % self.SHOW_EVERY == 0 and run != 0:
                    self.env.render()

            # monitor progress
            episodes_rewards.append(rew)

            if (run+1) % self.UPDATE_EVERY == 0:
                time_spent = datetime.datetime.now() - epoch_start_time
                epoch_start_time = datetime.datetime.now()
                avgsecs = total_secs / total_reward_in_epoch
                total_secs = 0
                updates_message = 'Run: {0} Average: {1} time spent {2} avg_time_iter {3}'.format(run,
                                                                                total_reward_in_epoch / self.UPDATE_EVERY,
                                                                                str(time_spent), avgsecs)
                logging.info(updates_message)
                print(updates_message)
                total_reward_in_epoch = 0

            if rew < 500:
                unsuccessful_episodes_count += 1

        logging.info(f'unsuccessful episodes => {unsuccessful_episodes_count}')
        base_file_name = f'_rewards_rsl-{self.RANDOM_START_LEVEL}_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}_init_{self.INITIAL_POLE_ANGLE}'
        file_path = f'./logs/cartpole/dqn/inference/{datetime.datetime.now()}_{base_file_name}.pkl'
        store_rewards(episodes_rewards, file_path)
        # show_fails_success_comparisson(self.RUNS, self.OBJECTIVE, episodes_rewards,
        #                                          self.RANDOM_START_LEVEL, self.RANDOM_PERTURBATIONS_LEVEL,
        #                                          self.PERTURBATIONS_INTENSITY_STD, self.INITIAL_POLE_ANGLE);
