from datetime import datetime, timedelta
import os
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesQlearnGazebo,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    render_params,
    save_dataframe_episodes,
    save_batch,
    save_best_episode,
    LoggingHandler,
)
from rl_studio.algorithms.qlearn import QLearn
from rl_studio.envs.gazebo.gazebo_envs import *


class TrainerFollowLineQlearnF1Gazebo:
    """
    Mode: training
    Task: Follow Line
    Algorithm: Qlearn
    Agent: F1
    Simulator: Gazebo
    """

    def __init__(self, config):
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesQlearnGazebo(config)
        self.global_params = LoadGlobalParams(config)

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"

    def main(self):

        log = LoggingHandler(self.log_file)

        ## Load Environment
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.environment.environment["epsilon"]
        epsilon_decay = epsilon / (self.env_params.total_episodes // 2)
        states_counter = {}

        log.logger.info(
            f"\nactions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"actions = {self.global_params.actions_set}\n"
            f"epsilon = {epsilon}\n"
            f"epsilon_decay = {epsilon_decay}\n"
            f"alpha = {self.environment.environment['alpha']}\n"
            f"gamma = {self.environment.environment['gamma']}\n"
        )
        ## --- init Qlearn
        qlearn = QLearn(
            actions=range(len(self.global_params.actions_set)),
            epsilon=self.environment.environment["epsilon"],
            alpha=self.environment.environment["alpha"],
            gamma=self.environment.environment["gamma"],
        )
        ## retraining q model
        if self.environment.environment["mode"] == "retraining":
            qlearn = qlearn.load_model(
                f"{self.global_params.models_dir}/{self.environment.environment['retrain_qlearn_model_name']}"
            )

        ## -------------    START TRAINING --------------------
        for episode in tqdm(
            range(1, self.env_params.total_episodes + 1), ascii=True, unit="episodes"
        ):
            done = False
            cumulated_reward = 0
            step = 0
            start_time_epoch = datetime.now()

            ## reset env()
            observation = env.reset()
            state = "".join(map(str, observation))

            while not done:
                step += 1
                # Pick an action based on the current state
                action = qlearn.selectAction(state)

                # Execute the action and get feedback
                observation, reward, done, _ = env.step(action, step)
                cumulated_reward += reward
                next_state = "".join(map(str, observation))
                qlearn.learn(state, action, reward, next_state)
                state = next_state

                # render params
                render_params(
                    action=action,
                    episode=episode,
                    step=step,
                    v=self.global_params.actions_set[action][
                        0
                    ],  # this case for discrete
                    w=self.global_params.actions_set[action][
                        1
                    ],  # this case for discrete
                    epsilon=self.environment.environment["epsilon"],
                    observation=observation,
                    reward_in_step=reward,
                    cumulated_reward=cumulated_reward,
                    done=done,
                )

                log.logger.debug(
                    f"\nepisode = {episode}\n"
                    f"step = {step}\n"
                    f"actions_len = {len(self.global_params.actions_set)}\n"
                    f"actions_range = {range(len(self.global_params.actions_set))}\n"
                    f"actions = {self.global_params.actions_set}\n"
                    f"epsilon = {epsilon}\n"
                    f"epsilon_decay = {epsilon_decay}\n"
                    f"v = {self.global_params.actions_set[action][0]}\n"
                    f"w = {self.global_params.actions_set[action][1]}\n"
                    f"observation = {observation}\n"
                    f"reward_in_step = {reward}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"done = {done}\n"
                )

                try:
                    states_counter[next_state] += 1
                except KeyError:
                    states_counter[next_state] = 1

                # best episode and step's stats
                if current_max_reward <= cumulated_reward and episode > 1:
                    (
                        current_max_reward,
                        best_epoch,
                        best_step,
                        best_epoch_training_time,
                    ) = save_best_episode(
                        self.global_params,
                        cumulated_reward,
                        episode,
                        step,
                        start_time_epoch,
                        reward,
                        env.image_center,
                    )

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step:
                    save_dataframe_episodes(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        self.global_params.aggr_ep_rewards,
                        self.global_params.actions_rewards,
                    )
                    log.logger.info(
                        f"saving batch of steps\n"
                        f"current_max_reward = {cumulated_reward}\n"
                        f"best_epoch = {best_epoch}\n"
                        f"best_step = {best_step}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )
                # End epoch
                if step > self.env_params.estimated_steps:
                    done = True
                    qlearn.save_model(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        qlearn,
                        cumulated_reward,
                        episode,
                        step,
                        epsilon,
                    )
                    log.logger.info(
                        f"\nEpisode COMPLETED\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epsilon = {epsilon}\n"
                    )

            # Save best lap
            if cumulated_reward >= current_max_reward:
                self.global_params.best_current_epoch["best_epoch"].append(best_epoch)
                self.global_params.best_current_epoch["highest_reward"].append(
                    cumulated_reward
                )
                self.global_params.best_current_epoch["best_step"].append(best_step)
                self.global_params.best_current_epoch[
                    "best_epoch_training_time"
                ].append(best_epoch_training_time)
                self.global_params.best_current_epoch[
                    "current_total_training_time"
                ].append(datetime.now() - start_time)
                save_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )
                qlearn.save_model(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    qlearn,
                    cumulated_reward,
                    episode,
                    step,
                    epsilon,
                )
                log.logger.info(
                    f"\nsaving best lap\n"
                    f"in episode = {episode}\n"
                    f"current_max_reward = {cumulated_reward}\n"
                    f"steps = {step}\n"
                    f"epsilon = {epsilon}\n"
                )
            # ended at training time setting: 2 hours, 15 hours...
            if (
                datetime.now() - timedelta(hours=self.global_params.training_time)
                > start_time
            ):
                if cumulated_reward >= current_max_reward:
                    save_dataframe_episodes(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        self.global_params.aggr_ep_rewards,
                    )
                    qlearn.save_model(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        qlearn,
                        cumulated_reward,
                        episode,
                        step,
                        epsilon,
                    )
                    log.logger.info(
                        f"\nTraining Time over\n"
                        f"current_max_reward = {cumulated_reward}\n"
                        f"epoch = {episode}\n"
                        f"step = {step}\n"
                        f"epsilon = {epsilon}\n"
                    )
                break

            # save best values every save_episode times
            self.global_params.ep_rewards.append(cumulated_reward)
            if not episode % self.env_params.save_episodes:
                save_batch(
                    episode,
                    step,
                    start_time_epoch,
                    start_time,
                    self.global_params,
                    self.env_params,
                )
                save_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )
                log.logger.info(
                    f"\nsaving BATCH\n"
                    f"current_max_reward = {cumulated_reward}\n"
                    f"best_epoch = {best_epoch}\n"
                    f"best_step = {best_step}\n"
                    f"best_epoch_training_time = {best_epoch_training_time}\n"
                )
            # updating epsilon for exploration
            if epsilon > self.environment.environment["epsilon_min"]:
                # self.epsilon *= self.epsilon_discount
                epsilon -= epsilon_decay
                epsilon = qlearn.updateEpsilon(
                    max(self.environment.environment["epsilon_min"], epsilon)
                )

        # save the last one
        save_dataframe_episodes(
            self.environment.environment,
            self.global_params.metrics_data_dir,
            self.global_params.aggr_ep_rewards,
        )
        env.close()
