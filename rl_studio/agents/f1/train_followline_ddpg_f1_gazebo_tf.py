from datetime import datetime, timedelta
import os
import pprint
import random
import time

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm


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
        ou_noise = OUActionNoise(
            mean=np.zeros(1),
            std_deviation=float(self.algoritmhs_params.std_dev) * np.ones(1),
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
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
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
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = ac_agent.policy(
                    tf_prev_state, ou_noise, self.global_params.actions
                )
                state, reward, done, _ = env.step(action, step)
                cumulated_reward += reward

                # learn and update
                buffer.record((prev_state, action, reward, state))
                buffer.learn(ac_agent, self.algoritmhs_params.gamma)
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
                    # v=action[0][0],  # for continuous actions
                    # w=action[0][1],  # for continuous actions
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
                    f"reward_in_step = {reward}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"done = {done}\n"
                )

                # best episode
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    best_epoch_training_time = datetime.now() - start_time_epoch
                    # saving params to show
                    self.global_params.actions_rewards["episode"].append(episode)
                    self.global_params.actions_rewards["step"].append(step)
                    # For continuous actios
                    # self.actions_rewards["v"].append(action[0][0])
                    # self.actions_rewards["w"].append(action[0][1])
                    self.global_params.actions_rewards["reward"].append(reward)
                    self.global_params.actions_rewards["center"].append(
                        env.image_center
                    )

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step:
                    save_dataframe_episodes(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        self.global_params.aggr_ep_rewards,
                        self.global_params.actions_rewards,
                    )
                    log.logger.debug(
                        f"SHOWING BATCH OF STEPS\n"
                        f"current_max_reward = {cumulated_reward}\n"
                        f"current epoch = {episode}\n"
                        f"current step = {step}\n"
                        f"best epoch so far = {best_epoch}\n"
                        f"best step so far = {best_step}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )

                #####################################################
                ### save in case of completed steps in one episode
                if step >= self.env_params.estimated_steps:
                    done = True
                    log.logger.info(
                        f"\nEPISODE COMPLETED\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                    )
                    save_actorcritic_model(
                        ac_agent,
                        self.global_params,
                        self.algoritmhs_params,
                        cumulated_reward,
                        episode,
                        "LAPCOMPLETED",
                    )
            #####################################################
            #### save best lap in episode
            if (
                cumulated_reward - self.environment.environment["rewards"]["penal"]
            ) >= current_max_reward and episode > 1:

                self.global_params.best_current_epoch["best_epoch"].append(best_epoch)
                self.global_params.best_current_epoch["highest_reward"].append(
                    current_max_reward
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
                    self.global_params.best_current_epoch,
                )
                save_actorcritic_model(
                    ac_agent,
                    self.global_params,
                    self.algoritmhs_params,
                    self.environment.environment,
                    cumulated_reward,
                    episode,
                    "BESTLAP",
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
                if cumulated_reward > current_max_reward:
                    save_actorcritic_model(
                        ac_agent,
                        self.global_params,
                        self.algoritmhs_params,
                        self.environment.environment,
                        cumulated_reward,
                        episode,
                        "FINISHTIME",
                    )

                break

            #####################################################
            ### save every save_episode times
            self.global_params.ep_rewards.append(cumulated_reward)
            if not episode % self.env_params.save_episodes:
                average_reward = sum(
                    self.global_params.ep_rewards[-self.env_params.save_episodes :]
                ) / len(self.global_params.ep_rewards[-self.env_params.save_episodes :])
                min_reward = min(
                    self.global_params.ep_rewards[-self.env_params.save_episodes :]
                )
                max_reward = max(
                    self.global_params.ep_rewards[-self.env_params.save_episodes :]
                )
                tensorboard.update_stats(
                    reward_avg=int(average_reward),
                    reward_max=int(max_reward),
                    steps=step,
                )
                self.global_params.aggr_ep_rewards["episode"].append(episode)
                self.global_params.aggr_ep_rewards["step"].append(step)
                self.global_params.aggr_ep_rewards["avg"].append(average_reward)
                self.global_params.aggr_ep_rewards["max"].append(max_reward)
                self.global_params.aggr_ep_rewards["min"].append(min_reward)
                self.global_params.aggr_ep_rewards["epoch_training_time"].append(
                    (datetime.now() - start_time_epoch).total_seconds()
                )
                self.global_params.aggr_ep_rewards["total_training_time"].append(
                    (datetime.now() - start_time).total_seconds()
                )
                save_actorcritic_model(
                    ac_agent,
                    self.global_params,
                    self.algoritmhs_params,
                    self.environment.environment,
                    cumulated_reward,
                    episode,
                    "BATCH",
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
        #####################################################
        ### save last episode, not neccesarily the best one
        save_dataframe_episodes(
            self.environment.environment,
            self.global_params.metrics_data_dir,
            self.global_params.aggr_ep_rewards,
        )
        env.close()
