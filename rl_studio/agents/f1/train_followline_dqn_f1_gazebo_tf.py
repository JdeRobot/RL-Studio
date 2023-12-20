from datetime import datetime, timedelta
import logging
import os
import pprint
import random
import time
import psutil

import gymnasium as gym

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesDQNGazebo,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    print_messages,
    print_dictionary,
    render_params,
    save_dataframe_episodes,
    save_batch,
    save_best_episode_dqn,
    LoggingHandler,
)
from rl_studio.algorithms.dqn_keras import (
    ModifiedTensorBoard,
    DQN,
)
from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO


class TrainerFollowLineDQNF1GazeboTF:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DQN
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesDQNGazebo(config)
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

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.algoritmhs_params.epsilon
        epsilon_discount = self.algoritmhs_params.epsilon_discount
        epsilon_min = self.algoritmhs_params.epsilon_min

        ## Reset env
        observation, state_size = env.reset()

        log.logger.info(
            f"\nstates = {self.global_params.states}\n"
            f"states_set = {self.global_params.states_set}\n"
            f"states_len = {len(self.global_params.states_set)}\n"
            f"actions = {self.global_params.actions}\n"
            f"actions set = {self.global_params.actions_set}\n"
            f"actions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"epsilon = {epsilon}\n"
            f"batch_size = {self.algoritmhs_params.batch_size}\n"
            f"logs_tensorboard_dir = {self.global_params.logs_tensorboard_dir}\n"
        )

        ## --------------------- Deep Nets ------------------
        # Init Agent
        dqn_agent = DQN(
            self.environment.environment,
            self.algoritmhs_params,
            len(self.global_params.actions_set),
            state_size,
            self.global_params.models_dir,
            self.global_params,
        )
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        ## -------------    START TRAINING --------------------
        for episode in tqdm(
            range(1, self.env_params.total_episodes + 1),
            ascii=True,
            unit="episodes",
        ):
            tensorboard.step = episode
            done = False
            cumulated_reward = 0
            step = 1
            start_time_epoch = datetime.now()

            observation, _ = env.reset()

            while not done:
                if np.random.random() > epsilon:
                    action = np.argmax(dqn_agent.get_qs(observation))
                else:
                    # Get random action
                    action = np.random.randint(0, len(self.global_params.actions_set))

                new_observation, reward, done, _ = env.step(action, step)

                # Every step we update replay memory and train main network
                # agent_dqn.update_replay_memory((state, action, reward, nextState, done))
                dqn_agent.update_replay_memory(
                    (observation, action, reward, new_observation, done)
                )

                cumulated_reward += reward
                observation = new_observation
                step += 1

                log.logger.debug(
                    f"\nobservation = {observation}\n"
                    # f"observation[0]= {observation[0]}\n"
                    f"observation type = {type(observation)}\n"
                    # f"observation[0] type = {type(observation[0])}\n"
                    f"new_observation = {new_observation}\n"
                    f"new_observation = {type(new_observation)}\n"
                    f"action = {action}\n"
                    f"actions type = {type(action)}\n"
                )
                render_params(
                    task=self.global_params.task,
                    episode=episode,
                    step=step,
                    state=observation,
                    action=action,
                    # v=self.global_params.actions_set[action][
                    #    0
                    # ],  # this case for discrete
                    # w=self.global_params.actions_set[action][
                    #    1
                    # ],  # this case for discrete
                    # self.env.image_center,
                    # self.actions_rewards,
                    reward_in_step=reward,
                    cumulated_reward_in_this_episode=cumulated_reward,
                    _="--------------------------",
                    best_episode_until_now=best_epoch,
                    in_best_step=best_step,
                    with_highest_reward=int(current_max_reward),
                    in_best_epoch_training_time=best_epoch_training_time,
                )

                log.logger.debug(
                    f"\nepisode = {episode}\n"
                    f"step = {step}\n"
                    f"actions_len = {len(self.global_params.actions_set)}\n"
                    f"actions_range = {range(len(self.global_params.actions_set))}\n"
                    f"actions = {self.global_params.actions_set}\n"
                    f"epsilon = {epsilon}\n"
                    # f"v = {self.global_params.actions_set[action][0]}\n"
                    # f"w = {self.global_params.actions_set[action][1]}\n"
                    f"observation = {observation}\n"
                    f"reward_in_step = {reward}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"done = {done}\n"
                )

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step:
                    log.logger.debug(
                        f"SHOWING BATCH OF STEPS\n"
                        f"current_max_reward = {cumulated_reward}\n"
                        f"current epoch = {episode}\n"
                        f"current step = {step}\n"
                        f"best epoch so far = {best_epoch}\n"
                        f"best step so far = {best_step}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )
                    dqn_agent.train(done, step)

                #####################################################
                ### save in case of completed steps in one episode
                if step >= self.env_params.estimated_steps:
                    done = True
                    log.logger.info(
                        f"\nEPISODE COMPLETED\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epsilon = {epsilon}\n"
                    )
                    dqn_agent.model.save(
                        f"{self.global_params.models_dir}/{self.algoritmhs_params.model_name}_LAPCOMPLETED_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )

            #####################################################
            #### save best lap in episode
            if cumulated_reward > current_max_reward and episode > 1:
                current_max_reward = cumulated_reward
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
                log.logger.info(
                    f"\nsaving best lap\n"
                    f"in episode = {episode}\n"
                    f"current_max_reward = {cumulated_reward}\n"
                    f"steps = {step}\n"
                    f"epsilon = {epsilon}\n"
                )
                dqn_agent.model.save(
                    f"{self.global_params.models_dir}/{self.algoritmhs_params.model_name}_IMPROVE_{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                )
                save_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )


            #####################################################
            ### end episode in time settings: 2 hours, 15 hours...
            if (
                datetime.now() - timedelta(hours=self.global_params.training_time)
                > start_time
            ):
                log.logger.info(
                    f"\nTraining Time over\n"
                    f"current_max_reward = {cumulated_reward}\n"
                    f"epoch = {episode}\n"
                    f"step = {step}\n"
                    f"epsilon = {epsilon}\n"
                )
                if cumulated_reward > current_max_reward:
                    dqn_agent.model.save(
                        f"{self.global_params.models_dir}/{self.algoritmhs_params.model_name}_LAPCOMPLETED_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )

                break

            #####################################################
            ### save every save_episode times
            self.global_params.ep_rewards.append(cumulated_reward)

            # average_reward = sum(
            #     self.global_params.ep_rewards[-self.env_params.save_episodes:]
            # ) / len(self.global_params.ep_rewards[-self.env_params.save_episodes:])
            average_reward = self.global_params.ep_rewards[-1]
            min_reward = min(
                self.global_params.ep_rewards[-self.env_params.save_episodes:]
            )
            max_reward = max(
                self.global_params.ep_rewards[-self.env_params.save_episodes:]
            )

            # Initialize lists to store timestamps and memory usage values
            timestamps = []
            memory_usage = []

            # Function to get current memory usage and log it
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            usage = psutil.virtual_memory().percent  # Get the percentage of used virtual memory
            timestamps.append(timestamp)
            memory_usage.append(usage)
            tensorboard.update_stats(
                reward_avg=int(average_reward),
                reward_max=int(max_reward),
                steps=step,
                epsilon=epsilon,
                memory=usage,
            )

            if not episode % self.env_params.save_episodes:
                self.global_params.aggr_ep_rewards["episode"].append(episode)
                self.global_params.aggr_ep_rewards["avg"].append(average_reward)
                self.global_params.aggr_ep_rewards["max"].append(max_reward)
                self.global_params.aggr_ep_rewards["min"].append(min_reward)
                self.global_params.aggr_ep_rewards["epoch_training_time"].append(
                    (datetime.now() - start_time_epoch).total_seconds()
                )

                dqn_agent.model.save(
                    f"{self.global_params.models_dir}/{self.algoritmhs_params.model_name}_BATCH_{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
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
            # reducing exploration
            if epsilon > epsilon_min:
                epsilon *= epsilon_discount

        #####################################################
        ### save last episode, not neccesarily the best one
        save_dataframe_episodes(
            self.environment.environment,
            self.global_params.metrics_data_dir,
            self.global_params.aggr_ep_rewards,
        )
        env.close()
