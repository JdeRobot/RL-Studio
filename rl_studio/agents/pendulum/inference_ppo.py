import datetime
import time
import random

import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils import tensorboard
from tqdm import tqdm
import numpy as np
import torch

import logging

from rl_studio.agents.pendulum import utils
from rl_studio.algorithms.ppo_continuous import PPO
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO
from rl_studio.agents.pendulum.utils import store_rewards, save_metadata
from rl_studio.wrappers.inference_rlstudio import InferencerWrapper


# # https://github.com/openai/gym/blob/master/gym/core.py
# class NormalizedEnv(gym.ActionWrapper):
#     """ Wrap action """
#
#     def _action(self, action):
#         act_k = (self.action_space.high - self.action_space.low) / 2.
#         act_b = (self.action_space.high + self.action_space.low) / 2.
#         return act_k * action + act_b
#
#     def _reverse_action(self, action):
#         act_k_inv = 2. / (self.action_space.high - self.action_space.low)
#         act_b = (self.action_space.high + self.action_space.low) / 2.
#         return act_k_inv * (action - act_b)


class PPOPendulumTrainer:
    def __init__(self, params):

        self.now = datetime.datetime.now()
        # self.environment params
        self.params = params
        self.environment_params = params["environments"]
        self.env_name = params["environments"]["env_name"]
        self.config = params["settings"]
        self.agent_config = params["agent"]

        if self.config["logging_level"] == "debug":
            self.LOGGING_LEVEL = logging.DEBUG
        elif self.config["logging_level"] == "error":
            self.LOGGING_LEVEL = logging.ERROR
        elif self.config["logging_level"] == "critical":
            self.LOGGING_LEVEL = logging.CRITICAL
        else:
            self.LOGGING_LEVEL = logging.INFO

        self.RANDOM_PERTURBATIONS_LEVEL = self.environment_params.get("random_perturbations_level", 0)
        self.PERTURBATIONS_INTENSITY_STD = self.environment_params.get("perturbations_intensity_std", 0)

        # Unfortunately, max_steps is not working with new_step_api=True and it is not giving any benefit.
        # self.env = gym.make(self.env_name, new_step_api=True, random_start_level=random_start_level)
        # self.env = NormalizedEnv(gym.make(self.env_name
        #                                   # ,random_start_level=self.RANDOM_START_LEVEL, initial_pole_angle=self.INITIAL_POLE_ANGLE,
        #                                   # non_recoverable_angle=non_recoverable_angle
        #                                   ))
        self.env = gym.make(self.env_name)
        self.RUNS = self.environment_params["runs"]
        self.UPDATE_EVERY = self.environment_params[
            "update_every"
        ]  # How often the current progress is recorded
        self.OBJECTIVE_REWARD = self.environment_params[
            "objective_reward"
        ]

        self.losses_list, self.reward_list, self.episode_len_list= (
            [],
            [],
            [],
        )  # metrics
        # recorded for graph
        self.episodes_update = params.get("algorithm").get("episodes_update")

        self.max_avg = -1000

        self.num_actions = self.env.action_space.shape[0]
        input_dim = self.env.observation_space.shape[0]
        lr_actor = 0.0003
        lr_critic = 0.001
        K_epochs = 80
        action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        self.action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

        inference_file = params["inference"]["inference_file"]
        self.inferencer = InferencerWrapper("ppo_continuous", inference_file, env=self.env)

    def print_init_info(self):
        logging.info(JDEROBOT)
        logging.info(JDEROBOT_LOGO)
        logging.info(f"\t- Start hour: {datetime.datetime.now()}\n")
        logging.info(f"\t- self.environment params:\n{self.environment_params}")

    def gather_statistics(self, losses, ep_len, episode_rew):
        if losses is not None:
            self.losses_list.append(losses / ep_len)
        self.reward_list.append(episode_rew)
        self.episode_len_list.append(ep_len)

    def main(self):
        epoch_start_time = datetime.datetime.now()

        logs_dir = 'logs/pendulum/ppo/inference/'
        logs_file_name = 'logs_file_' + str(
            self.RANDOM_PERTURBATIONS_LEVEL) + '_' + str(epoch_start_time) \
                         + str(self.PERTURBATIONS_INTENSITY_STD) + '.log'
        logging.basicConfig(filename=logs_dir + logs_file_name, filemode='a',
                            level=self.LOGGING_LEVEL,
                            format='%(name)s - %(levelname)s - %(message)s')
        self.print_init_info()

        start_time_format = epoch_start_time.strftime("%Y%m%d_%H%M")

        logging.info(LETS_GO)
        w = tensorboard.SummaryWriter(log_dir=f"{logs_dir}/tensorboard/{start_time_format}")

        actor_loss = 0
        critic_loss = 0
        total_reward_in_epoch = 0
        global_steps = 0

        for episode in tqdm(range(self.RUNS)):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            while not done:
                step += 1
                global_steps += 1
                # if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                #     perturbation_action = random.randrange(self.env.action_space.n)
                #     state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                #     logging.debug("perturbated in step {} with action {}".format(episode_rew, perturbation_action))

                action = self.inferencer.inference(state)
                new_state, reward, _, done, _ = self.env.step(action)

                state = new_state
                episode_reward += reward
                total_reward_in_epoch += reward

                w.add_scalar("reward/episode_reward", episode_reward, global_step=episode)

            self.gather_statistics(actor_loss, step, episode_reward)

            # monitor progress
            if (episode + 1) % self.UPDATE_EVERY == 0:
                time_spent = datetime.datetime.now() - epoch_start_time
                epoch_start_time = datetime.datetime.now()
                updates_message = 'Run: {0} Average: {1} time spent {2}'.format(episode,
                                                                                total_reward_in_epoch / self.UPDATE_EVERY,
                                                                                str(time_spent))
                logging.info(updates_message)
                print(updates_message)
                last_average = total_reward_in_epoch / self.UPDATE_EVERY;

                if last_average >= self.OBJECTIVE_REWARD:
                    logging.info("Training objective reached!!")
                    break
                total_reward_in_epoch = 0

        base_file_name = f'_rewards_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}'
        file_path = f'{logs_dir}{datetime.datetime.now()}_{base_file_name}.pkl'
        store_rewards(self.reward_list, file_path)
        plt.plot(self.reward_list)
        plt.legend("reward per episode")
        plt.show()
