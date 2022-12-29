import datetime
import time
import random

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import logging

from rl_studio.agents.cartpole import utils
from rl_studio.algorithms.dqn_torch import DQN_Agent
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO
from rl_studio.agents.cartpole.utils import store_rewards, save_metadata


class DQNCartpoleTrainer:
    def __init__(self, params):

        self.now = datetime.datetime.now()
        # self.environment params
        self.params = params
        self.environment_params = params["environments"]
        self.env_name = self.environment_params["env_name"]
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
        self.RANDOM_START_LEVEL = self.environment_params.get("random_start_level", 0)
        self.INITIAL_POLE_ANGLE = self.environment_params.get("initial_pole_angle", None)


        non_recoverable_angle = self.environment_params[
            "non_recoverable_angle"
        ]
        # Unfortunately, max_steps is not working with new_step_api=True and it is not giving any benefit.
        # self.env = gym.make(self.env_name, new_step_api=True, random_start_level=random_start_level)
        self.env = gym.make(self.env_name, random_start_level=self.RANDOM_START_LEVEL, initial_pole_angle=self.INITIAL_POLE_ANGLE,
                            non_recoverable_angle=non_recoverable_angle)

        self.RUNS = self.environment_params["runs"]
        self.EXPERIMENTATION_RUNS = self.environment_params["full_experimentation_runs"]
        self.SHOW_EVERY = self.environment_params[
            "show_every"
        ]
        self.UPDATE_EVERY = self.environment_params[
            "update_every"
        ]  # How often the current progress is recorded
        self.OBJECTIVE_REWARD = self.environment_params[
            "objective_reward"
        ]
        self.BLOCKED_EXPERIENCE_BATCH = self.environment_params[
            "block_experience_batch"
        ]

        self.actions = self.env.action_space.n

        self.losses_list, self.reward_list, self.episode_len_list, self.epsilon_list = (
            [],
            [],
            [],
            [],
        )  # metrics
        # recorded for graph
        self.epsilon = 1
        self.EPSILON_DISCOUNT = params["algorithm"]["epsilon_discount"]
        self.GAMMA = params["algorithm"]["gamma"]
        self.NUMBER_OF_EXPLORATION_STEPS = 128

        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.exp_replay_size = params["algorithm"]["batch_size"]
        self.deepq = DQN_Agent(
            layer_sizes=[input_dim, 64, output_dim],
            lr=1e-3,
            sync_freq=5,
            exp_replay_size=self.exp_replay_size,
            seed=1423,
            gamma=self.GAMMA,
            block_batch=self.BLOCKED_EXPERIENCE_BATCH
        )
        self.max_avg = 0
        self.initialize_experience_replay()

    def initialize_experience_replay(self):
        index = 0
        for i in range(self.exp_replay_size):
            state = self.env.reset()
            done = False
            while not done:
                A = self.deepq.get_action(state, self.env.action_space.n, epsilon=1)
                next_state, reward, done, _ = self.env.step(A.item())
                self.deepq.collect_experience([state, A.item(), reward, next_state])
                state = next_state
                index += 1
                if index > self.exp_replay_size:
                    break

    def print_init_info(self):
        logging.info(JDEROBOT)
        logging.info(JDEROBOT_LOGO)
        logging.info(f"\t- Start hour: {datetime.datetime.now()}\n")
        logging.info(f"\t- self.environment params:\n{self.environment_params}")

    def evaluate_and_collect(self, state):
        A = self.deepq.get_action(state, self.env.action_space.n, self.epsilon)
        next_state, reward, done, _ = self.env.step(A.item())
        self.deepq.collect_experience([state, A.item(), reward, next_state])

        return next_state, reward, done

    def train_in_batches(self, trainings, batch_size):
        losses = 0
        for j in range(trainings):
            loss = self.deepq.train(batch_size=batch_size)
            losses += loss
        return losses

    def gather_statistics(self, losses, ep_len, episode_rew):
        self.losses_list.append(losses / ep_len)
        self.reward_list.append(episode_rew)
        self.episode_len_list.append(ep_len)
        self.epsilon_list.append(self.epsilon)

    # def final_demonstration(self):
    #     for i in tqdm(range(2)):
    #         obs, done, rew = self.env.reset(), False, 0
    #         while not done:
    #             obs = np.append(obs, -1)
    #             A = self.deepq.get_action(obs, self.env.action_space.n, epsilon=0)
    #             obs, reward, done, info = self.env.step(A.item())
    #             rew += reward
    #             time.sleep(0.01)
    #             self.env.render()
    #         logging.info("\ndemonstration episode : {}, reward : {}".format(i, rew))

    def main(self):
        epoch_start_time = datetime.datetime.now()

        logs_dir = 'logs/cartpole/dqn/training/'
        logs_file_name = 'logs_file_' + str(self.RANDOM_START_LEVEL) + '_' + str(
            self.RANDOM_PERTURBATIONS_LEVEL) + '_' + str(epoch_start_time) \
                         + str(self.PERTURBATIONS_INTENSITY_STD) + '.log'
        logging.basicConfig(filename=logs_dir + logs_file_name, filemode='a',
                            level=self.LOGGING_LEVEL,
                            format='%(name)s - %(levelname)s - %(message)s')
        self.print_init_info()

        start_time_format = epoch_start_time.strftime("%Y%m%d_%H%M")

        if self.config["save_model"]:
            save_metadata("dqn", start_time_format, self.params)

        logging.info(LETS_GO)
        number_of_steps = 128
        total_reward_in_epoch = 0
        for run in tqdm(range(self.RUNS)):
            state, done, losses, ep_len, episode_rew = self.env.reset(), False, 0, 0, 0
            while not done:
                ep_len += 1
                number_of_steps += 1
                if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                    perturbation_action = random.randrange(self.env.action_space.n)
                    state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                    logging.debug("perturbated in step {} with action {}".format(episode_rew, perturbation_action))
                next_state, reward, done = self.evaluate_and_collect(state)
                state = next_state
                episode_rew += reward
                total_reward_in_epoch += reward

                if run % self.SHOW_EVERY == 0:
                    self.env.render()
                if number_of_steps > self.NUMBER_OF_EXPLORATION_STEPS:
                    number_of_steps = 0
                    losses += self.train_in_batches(4, 16)

            if run > self.EXPERIMENTATION_RUNS and self.epsilon > 0.05:
                self.epsilon *= self.EPSILON_DISCOUNT

            self.gather_statistics(losses, ep_len, episode_rew)

            # monitor progress
            if (run+1) % self.UPDATE_EVERY == 0:
                time_spent = datetime.datetime.now() - epoch_start_time
                epoch_start_time = datetime.datetime.now()
                updates_message = 'Run: {0} Average: {1} epsilon {2} time spent {3}'.format(run, total_reward_in_epoch / self.UPDATE_EVERY,
                                                                                     self.epsilon, str(time_spent))
                logging.info(updates_message)
                print(updates_message)
                if self.config["save_model"] and total_reward_in_epoch / self.UPDATE_EVERY > self.max_avg:
                    self.max_avg = total_reward_in_epoch / self.UPDATE_EVERY
                    logging.info(f"Saving model . . .")
                    utils.save_dqn_model(self.deepq, start_time_format, total_reward_in_epoch / self.UPDATE_EVERY, self.params)
                if (total_reward_in_epoch / self.UPDATE_EVERY) > self.OBJECTIVE_REWARD:
                    logging.info("Training objective reached!!")
                    break
                total_reward_in_epoch = 0

        # self.final_demonstration()
        base_file_name = f'_rewards_rsl-{self.RANDOM_START_LEVEL}_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}'
        file_path = f'./logs/cartpole/dqn/training/{datetime.datetime.now()}_{base_file_name}.pkl'
        store_rewards(self.reward_list, file_path)
        plt.plot(self.reward_list)
        plt.legend("reward per episode")
        plt.show()
