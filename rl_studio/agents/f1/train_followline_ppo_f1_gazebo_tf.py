from datetime import datetime, timedelta
import os
import pprint
import random
import time

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from rl_studio.agents.utilities.plot_npy_dataset import plot_rewards
from rl_studio.agents.utilities.push_git_repo import git_add_commit_push

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesPPOGazebo,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    print_messages,
    render_params,
    save_dataframe_episodes,
    LoggingHandler,
)
from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)

from rl_studio.algorithms.ppo_continuous import PPO

from rl_studio.algorithms.utils import (
    save_actorcritic_model,
)
from rl_studio.envs.gazebo.gazebo_envs import *


def combine_attributes(obj1, obj2, obj3):
    combined_dict = {}

    # Extract attributes from obj1
    obj1_dict = obj1.__dict__
    for key, value in obj1_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj2
    obj2_dict = obj2.__dict__
    for key, value in obj2_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj3
    obj3_dict = obj3.__dict__
    for key, value in obj3_dict.items():
        combined_dict[key] = value

    return combined_dict


class TrainerFollowLinePPOF1GazeboTF:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DDPG
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesPPOGazebo(config)
        self.global_params = LoadGlobalParams(config)

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)
        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        # self.outdir = f"{self.global_params.models_dir}/ddpg/{self.global_params.states}"

    def main(self):

        log = LoggingHandler(self.log_file)
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        hyperparams = combine_attributes(self.algoritmhs_params,
                                         self.environment,
                                         self.global_params)

        tensorboard.update_hyperparams(hyperparams)

        ## Load Environment
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_epoch_training_time = 0
        all_steps = 0
        ## Reset env
        K_epochs = 80
        _, state_size = env.reset()

        self.ppo_agent = PPO(state_size, len(self.global_params.actions_set), self.algoritmhs_params.actor_lr,
                             self.algoritmhs_params.critic_lr, self.algoritmhs_params.gamma,
                             K_epochs, self.algoritmhs_params.epsilon)

        log.logger.info(
            f"\nstates = {self.global_params.states}\n"
            f"states_set = {self.global_params.states_set}\n"
            f"states_len = {len(self.global_params.states_set)}\n"
            f"actions = {self.global_params.actions}\n"
            f"actions set = {self.global_params.actions_set}\n"
            f"actions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"batch_size = {self.algoritmhs_params.batch_size}\n"
            f"logs_tensorboard_dir = {self.global_params.logs_tensorboard_dir}\n"
        )

        ## -------------    START TRAINING --------------------
        for episode in tqdm(
                range(1, self.env_params.total_episodes + 1), ascii=True, unit="episodes"
        ):
            tensorboard.step = episode
            done = False
            cumulated_reward = 0
            step = 1
            start_time_epoch = datetime.now()

            prev_state, prev_state_size = env.reset()

            while not done:
                all_steps += 1

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.ppo_agent.select_action(tf_prev_state)
                action[0] = action[0] + 1 * 4 # TODO scale it propperly
                tensorboard.update_actions(action, all_steps)

                state, reward, done, info = env.step([action], step)
                # fps = info["fps"]
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)

                # update PPO agent
                if all_steps % self.algoritmhs_params.episodes_update == 0:
                    self.ppo_agent.update()

                if all_steps % self.global_params.steps_to_decrease == 0:
                    self.ppo_agent.decay_action_std(self.global_params.decrease_substraction,
                                                    self.global_params.decrease_min)

                cumulated_reward += reward

                if self.global_params.show_monitoring:
                    log.logger.debug(
                        f"\nstate = {state}\n"
                        # f"observation[0]= {observation[0]}\n"
                        f"state type = {type(state)}\n"
                        # f"observation[0] type = {type(observation[0])}\n"
                        f"prev_state = {prev_state}\n"
                        f"prev_state = {type(prev_state)}\n"
                        f"action = {action}\n"
                        f"actions type = {type(action)}\n"
                        f"\nepisode = {episode}\n"
                        f"step = {step}\n"
                        f"actions_len = {len(self.global_params.actions_set)}\n"
                        f"actions_range = {range(len(self.global_params.actions_set))}\n"
                        f"actions = {self.global_params.actions_set}\n"
                        f"reward_in_step = {reward}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"done = {done}\n"
                    )
                    render_params(
                        task=self.global_params.task,
                        v=action[0],  # for continuous actions
                        w=action[1],  # for continuous actions
                        episode=episode,
                        step=step,
                        state=state,
                        # v=self.global_params.actions_set[action][
                        #    0
                        # ],  # this case for discrete
                        # w=self.global_params.actions_set[action][
                        #    1
                        # ],  # this case for discrete
                        reward_in_step=reward,
                        cumulated_reward_in_this_episode=cumulated_reward,
                        _="--------------------------",
                        # fps=fps,
                        exploration=self.ppo_agent.action_std,
                        best_episode_until_now=best_epoch,
                        with_highest_reward=int(current_max_reward),
                        in_best_epoch_training_time=best_epoch_training_time,
                    )

                prev_state = state
                step += 1

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not all_steps % self.env_params.save_every_step:
                    file_name = save_dataframe_episodes(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        self.global_params.aggr_ep_rewards,
                        self.global_params.actions_rewards,
                    )
                    plot_rewards(
                        self.global_params.metrics_data_dir,
                        file_name
                    )
                    git_add_commit_push("automatic_rewards_update")
                    log.logger.debug(
                        f"SHOWING BATCH OF STEPS\n"
                        f"current_max_reward = {cumulated_reward}\n"
                        f"current epoch = {episode}\n"
                        f"current step = {step}\n"
                        f"best epoch so far = {best_epoch}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )
                #####################################################
                ### save in case of completed steps in one episode
                if step >= self.env_params.estimated_steps:
                    done = True
                    print_messages(
                        "Lap completed in:",
                        time=datetime.now() - start_time_epoch,
                        in_episode=episode,
                        episode_reward=int(cumulated_reward),
                        with_steps=step,
                    )
                    self.ppo_agent.save(
                        f"{self.global_params.models_dir}/"
                        f"{time.strftime('%Y%m%d-%H%M%S')}_LAPCOMPLETED"
                        f"MaxReward-{int(cumulated_reward)}_"
                        f"Epoch-{episode}")

                    # save_agent_physics(
                    #     self.environment, self.outdir, self.actions_rewards, start_time
                    # )

            #####################################################
            #### save best lap in episode
            if current_max_reward <= cumulated_reward:
                current_max_reward = cumulated_reward
                best_epoch = episode
                # best_epoch_training_time = datetime.now() - start_time_epoch
                # # saving params to show
                # self.global_params.actions_rewards["episode"].append(episode)
                # self.global_params.actions_rewards["step"].append(step)
                # # For continuous actions
                # # self.actions_rewards["v"].append(action[0][0])
                # # self.actions_rewards["w"].append(action[0][1])
                # self.global_params.actions_rewards["reward"].append(reward)
                # self.global_params.actions_rewards["center"].append(
                #     env.image_center
                # )
                # self.global_params.best_current_epoch["best_epoch"].append(best_epoch)
                # self.global_params.best_current_epoch["highest_reward"].append(
                #     current_max_reward
                # )
                # self.global_params.best_current_epoch[
                #     "best_epoch_training_time"
                # ].append(best_epoch_training_time)
                # self.global_params.best_current_epoch[
                #     "current_total_training_time"
                # ].append(datetime.now() - start_time)

                # save_dataframe_episodes(
                #     self.environment.environment,
                #     self.global_params.metrics_data_dir,
                #     self.global_params.best_current_epoch,
                # )
                self.ppo_agent.save(
                    f"{self.global_params.models_dir}/"
                    f"{time.strftime('%Y%m%d-%H%M%S')}-IMPROVED"
                    f"MaxReward-{int(cumulated_reward)}_"
                    f"Epoch-{episode}")

                log.logger.info(
                    f"\nsaving best lap\n"
                    f"in episode = {episode}\n"
                    f"current_max_reward = {cumulated_reward}\n"
                    f"steps = {step}\n"
                )

            #####################################################
            ### end episode in time settings: 2 hours, 15 hours...
            if (
                    datetime.now() - timedelta(hours=self.global_params.training_time)
                    > start_time
            ) or (episode > self.env_params.total_episodes):
                log.logger.info(
                    f"\nTraining Time over\n"
                    f"current_max_reward = {cumulated_reward}\n"
                    f"epoch = {episode}\n"
                    f"step = {step}\n"
                )
                # if cumulated_reward > current_max_reward:
                # save_actorcritic_model(
                #     ac_agent,
                #     self.global_params,
                #     self.algoritmhs_params,
                #     cumulated_reward,
                #     episode,
                #     "FINISHTIME",
                # )

                break

        #####################################################
        ### save last episode, not neccesarily the best one
        save_dataframe_episodes(
            self.environment.environment,
            self.global_params.metrics_data_dir,
            self.global_params.aggr_ep_rewards,
        )
        env.close()
