from datetime import datetime, timedelta
import os
import random
import time

import gymnasium as gym

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from rl_studio.agents.utils import (
    print_messages,
    render_params,
    save_dataframe_episodes,
)
from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesDDPGGazebo,
    LoadGlobalParams,
)

from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
    OUActionNoise,
    Buffer,
    DDPGAgent,
)
from rl_studio.algorithms.utils import save_actorcritic_model
from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO


class InferencerFollowLaneDDPGF1GazeboTF:
    """
    Mode: Inference
    Task: Follow Lane
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

    def main(self):
        print_messages(
            "InferencerFollowLaneDDPGF1GazeboTF",
            # algoritmhs_params=self.algoritmhs_params,
            # env_params=self.env_params,
            environment=self.environment.environment,
            environment_rewards=self.environment.environment["rewards"],
            # global_params=self.global_params,
            global_params_models_dir=self.global_params.models_dir,
            global_params_logs_tensorboard_dir=self.global_params.logs_tensorboard_dir,
            global_params_metrics_data_dir=self.global_params.metrics_data_dir,
            global_params_metrics_graphics_dir=self.global_params.metrics_graphics_dir,
            # global_params_actions=self.global_params.actions,
            # env_params_env_name=self.env_params.env_name
            # config=config,
        )

        ## Load Environment
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_tensorboard_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)
        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        ## Reset env
        state, state_size = env.reset()

        # Print state and actions for Debug
        print_messages(
            "In InferencerFollowLaneDDPGF1GazeboTF",
            state_size=state_size,
            action_space=self.global_params.actions,
            action_size=len(self.global_params.actions_set),
        )
        print_messages(
            "InferencerFollowLaneDDPGF1GazeboTF()",
            std_deviation=float(self.algoritmhs_params.std_dev),
            action_size=len(self.global_params.actions_set),
            logs_tensorboard_dir=self.global_params.logs_tensorboard_dir,
            states=self.global_params.states,
            action_space=self.global_params.actions,
            buffer_capacity=self.algoritmhs_params.buffer_capacity,
            batch_size=self.algoritmhs_params.batch_size,
        )

        ## --------------------- Deep Nets ------------------
        # ou_noise = OUActionNoise(
        #    mean=np.zeros(1),
        #    std_deviation=float(self.algoritmhs_params.std_dev) * np.ones(1),
        # )
        # Init Agents
        ac_agent = DDPGAgent(
            self.environment.environment,
            len(self.global_params.actions_set),
            state_size,
            self.global_params.models_dir,
        )
        ## ----- Load model -----
        model = ac_agent.load_inference_model(
            self.global_params.models_dir, self.environment.environment
        )
        # init Buffer
        # buffer = Buffer(
        #    state_size,
        #    len(self.global_params.actions_set),
        #    self.global_params.states,
        #    self.global_params.actions,
        #    self.algoritmhs_params.buffer_capacity,
        #    self.algoritmhs_params.batch_size,
        # )
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        # show rewards stats per episode

        ## -------------    START INFERENCING --------------------
        print(LETS_GO)
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
                # action = ac_agent.policy(
                #    tf_prev_state, ou_noise, self.global_params.actions
                # )
                actions = model.predict(tf_prev_state)
                action = [[actions[0][0][0], actions[1][0][0]]]
                print_messages(
                    "inference",
                    action=action,
                    action_0=action[0],
                    action_0_0=action[0][0],
                    action_0_1=action[0][1],
                )

                state, reward, done, _ = env.step(action, step)
                cumulated_reward += reward

                """ NO UPDATES
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
                """
                #
                prev_state = state
                step += 1

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

                render_params(
                    task=self.global_params.task,
                    # v=action[0][0], # for continuous actions
                    # w=action[0][1], # for continuous actions
                    episode=episode,
                    step=step,
                    state=state,
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

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step:
                    save_dataframe_episodes(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        self.global_params.aggr_ep_rewards,
                        self.global_params.actions_rewards,
                    )
                #####################################################
                ### save in case of completed steps in one episode
                if step >= self.env_params.estimated_steps:
                    done = True
                    print_messages(
                        "Lap completed in:",
                        time=datetime.now() - start_time_epoch,
                        in_episode=episode,
                        episode_reward_in_this_episode=int(cumulated_reward),
                        with_steps=step,
                        best_episode_in_training=best_epoch,
                        in_best_step=best_step,
                        with_highest_reward=int(current_max_reward),
                    )

            #####################################################
            #### save best lap in episode
            if (
                cumulated_reward - self.environment.environment["rewards"]["penal"]
            ) >= current_max_reward and episode > 1:
                print_messages(
                    "Saving best lap",
                    best_episode_until_now=best_epoch,
                    in_best_step=best_step,
                    with_highest_reward=int(current_max_reward),
                    in_best_epoch_training_time=best_epoch_training_time,
                    total_training_time=(datetime.now() - start_time),
                )
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

            #####################################################
            ### end episode in time settings: 2 hours, 15 hours...
            if (
                datetime.now() - timedelta(hours=self.global_params.training_time)
                > start_time
            ):
                print_messages(
                    "Training time finished in:",
                    time=datetime.now() - start_time,
                    episode=episode,
                    cumulated_reward=cumulated_reward,
                    current_max_reward=current_max_reward,
                    total_time=(
                        datetime.now()
                        - timedelta(hours=self.global_params.training_time)
                    ),
                    best_episode_in_total_training=best_epoch,
                    in_the_very_best_step=best_step,
                    with_the_highest_Total_reward=int(current_max_reward),
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

                print_messages(
                    "Showing batch:",
                    current_episode_batch=episode,
                    max_reward_in_current_batch=int(max_reward),
                    best_epoch_in_all_training=best_epoch,
                    highest_reward_in_all_training=int(current_max_reward),
                    in_best_step=best_step,
                    total_time=(datetime.now() - start_time),
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

                save_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )

        #####################################################
        ### save last episode, not neccesarily the best one
        save_dataframe_episodes(
            self.environment.environment,
            self.global_params.metrics_data_dir,
            self.global_params.aggr_ep_rewards,
        )
        env.close()
