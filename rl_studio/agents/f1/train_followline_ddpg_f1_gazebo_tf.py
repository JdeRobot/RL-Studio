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
from  rl_studio.agents.utilities.push_git_repo import git_add_commit_push

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesDDPGGazebo,
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
    OUActionNoise,
    Buffer,
    DDPGAgent,
)
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


class TrainerFollowLineDDPGF1GazeboTF:
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
        self.environment = LoadEnvVariablesDDPGGazebo(config)
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

        actor_loss = 0
        critic_loss = 0
        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_epoch_training_time = 0
        all_steps = 0
        ## Reset env
        state, state_size = env.reset()

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

        ## --------------------- Deep Nets ------------------
        exploration = self.algoritmhs_params.std_dev
        ou_noise = OUActionNoise(
            mean=np.zeros(1),
            std_deviation=float(exploration) * np.ones(1),
        )
        # Init Agents
        ac_agent = DDPGAgent(
            self.environment.environment,
            len(self.global_params.actions_set),
            state_size,
            self.global_params.models_dir,
        )
        # init Buffer
        buffer = Buffer(
            state_size,
            len(self.global_params.actions_set),
            self.global_params.states,
            self.global_params.actions,
            self.algoritmhs_params.buffer_capacity,
            self.algoritmhs_params.batch_size,
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
                if not all_steps % self.global_params.steps_to_decrease:
                    exploration = max(self.global_params.decrease_min, exploration - self.global_params.decrease_substraction)
                    log.logger.debug("decreasing exploration to exploration")
                    ou_noise = OUActionNoise(
                        mean=np.zeros(1),
                        std_deviation=float(exploration) * np.ones(1),
                    )

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = ac_agent.policy(
                    tf_prev_state, ou_noise, self.global_params.actions
                )
                tensorboard.update_actions(action[0], all_steps)
                state, reward, done, info = env.step(action, step)
                fps = info["fps"]
                # print(fps)
                cumulated_reward += reward

                # learn and update
                # env._gazebo_pause()

                buffer.record((prev_state, action, reward, state))
                actor_loss, critic_loss = buffer.learn(ac_agent, self.algoritmhs_params.gamma)
                ac_agent.update_target(
                    ac_agent.target_actor.variables,
                    ac_agent.actor_model.variables,
                    self.algoritmhs_params.tau,
                )
                ac_agent.update_target(
                    ac_agent.target_critic.variables,
                    ac_agent.critic_model.variables,
                    self.algoritmhs_params.tau,
                )

                #
                prev_state = state
                step += 1

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
                    )
                    render_params(
                        task=self.global_params.task,
                        v=action[0][0],  # for continuous actions
                        w=action[0][1],  # for continuous actions
                        episode=episode,
                        step=step,
                        state=state,
                        reward_in_step=reward,
                        cumulated_reward_in_this_episode=cumulated_reward,
                        _="--------------------------",
                        # fps=fps,
                        exploration=exploration,
                        best_episode_until_now=best_epoch,
                        with_highest_reward=int(current_max_reward),
                        in_best_epoch_training_time=best_epoch_training_time,
                    )
                    log.logger.debug(
                        f"\nepisode = {episode}\n"
                        f"step = {step}\n"
                        f"actions_len = {len(self.global_params.actions_set)}\n"
                        f"actions_range = {range(len(self.global_params.actions_set))}\n"
                        f"actions = {self.global_params.actions_set}\n"
                        f"reward_in_step = {reward}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"done = {done}\n"
                    )

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not all_steps % self.env_params.save_every_step:
                    file_name = save_dataframe_episodes(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        self.global_params.aggr_ep_rewards)
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
                    save_actorcritic_model(
                        ac_agent,
                        self.global_params,
                        self.algoritmhs_params,
                        self.environment.environment,
                        current_max_reward,
                        episode,
                        "LAPCOMPLETED",
                    )



            #####################################################
            #### save best lap in episode
            if current_max_reward <= cumulated_reward:
                current_max_reward = cumulated_reward
                best_epoch = episode

                save_actorcritic_model(
                    ac_agent,
                    self.global_params,
                    self.algoritmhs_params,
                    self.environment.environment,
                    current_max_reward,
                    episode,
                    "IMPROVED",
                )
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
                break

            #####################################################
            ### save every save_episode times
            self.global_params.ep_rewards.append(cumulated_reward)
            if not episode % self.env_params.save_episodes:
                average_reward = sum(self.global_params.ep_rewards[-self.env_params.save_episodes:]) / len(
                    self.global_params.ep_rewards[-self.env_params.save_episodes:]
                )
                min_reward = min(self.global_params.ep_rewards[-self.env_params.save_episodes:])
                max_reward = max(self.global_params.ep_rewards[-self.env_params.save_episodes:])
                tensorboard.update_stats(
                    cum_rewards=average_reward,
                    reward_min=min_reward,
                    reward_max=max_reward,
                    actor_loss=actor_loss,
                    critic_loss=critic_loss
                )

                self.global_params.aggr_ep_rewards["episode"].append(episode)
                self.global_params.aggr_ep_rewards["avg"].append(average_reward)
                self.global_params.aggr_ep_rewards["max"].append(max_reward)
                self.global_params.aggr_ep_rewards["min"].append(min_reward)
                self.global_params.aggr_ep_rewards["epoch_training_time"].append(
                    (datetime.now() - start_time_epoch).total_seconds()
                )
        env.close()
