from collections import Counter, OrderedDict
from datetime import datetime, timedelta
import glob
from statistics import median
import os
import time

import carla
import cv2
import gymnasium as gym
import numpy as np
import pygame
from reloading import reloading
import tensorflow as tf
from tqdm import tqdm

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesDDPGCarla,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    render_params,
    render_params_left_bottom,
    save_dataframe_episodes,
    save_carla_dataframe_episodes,
    save_batch,
    save_best_episode,
    LoggingHandler,
    print_messages,
)
from rl_studio.agents.utilities.plot_stats import MetricsPlot, StatsDataFrame
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
from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.logger import logger
from rl_studio.envs.carla.utils.manual_control import HUD, World
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
)
from rl_studio.envs.carla.utils.synchronous_mode import (
    CarlaSyncMode,
    draw_image,
    get_font,
    should_quit,
)
from rl_studio.envs.carla.carla_env import CarlaEnv

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass


class TrainerFollowLaneDDPGAutoCarlaTF:
    """
    Mode: training
    Task: Follow Lane
    Algorithm: DDPG
    Agent: Auto
    Simulator: Carla
    Weather: Static
    Traffic: No

    The most simplest environment
    """

    def __init__(self, config):
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesDDPGCarla(config)
        self.global_params = LoadGlobalParams(config)

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.recorders_carla_dir}", exist_ok=True)

        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"

        # print(f"\nin TrainerFollowLaneQlearnAutoCarla {config=}")
        # print(f"\nin TrainerFollowLaneQlearnAutoCarla {self.environment=}\n")
        # print(
        #    f"\nin TrainerFollowLaneQlearnAutoCarla {self.environment.environment=}\n"
        # )
        # lanzamos Carla server
        CarlaEnv.__init__(self)

    def main(self):
        """
        DDPG
        """
        log = LoggingHandler(self.log_file)
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

        print_messages(
            "main()",
            states=self.global_params.states,
            states_set=self.global_params.states_set,
            states_len=len(self.global_params.states_set),
            state_size=state_size,
            actions=self.global_params.actions,
            actions_set=self.global_params.actions_set,
            actions_len=len(self.global_params.actions_set),
            actions_range=range(len(self.global_params.actions_set)),
            batch_size=self.algoritmhs_params.batch_size,
            logs_tensorboard_dir=self.global_params.logs_tensorboard_dir,
            rewards=self.environment.environment["rewards"],
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
            range(1, self.env_params.total_episodes + 1),
            ascii=True,
            unit="episodes",
        ):
            tensorboard.step = episode
            # time.sleep(0.1)
            done = False
            cumulated_reward = 0
            step = 1
            start_time_epoch = time.time()  # datetime.now()

            prev_state, prev_state_size = env.reset()

            while not done:
                # if self.environment.environment["sync"]:
                env.world.tick()
                # else:
                #    env.world.wait_for_tick()
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = ac_agent.policy(
                    tf_prev_state, ou_noise, self.global_params.actions
                )
                start_step = time.time()
                state, reward, done, _ = env.step(action)
                end_step = time.time()

                self.global_params.time_steps[step] = end_step - start_step

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

                """
                print_messages(
                    "",
                    episode=episode,
                    step=step,
                    action=action,
                    v=action[0][0],  # for continuous actions
                    w=action[0][1],  # for continuous actions
                    state=state,
                    reward=reward,
                    done=done,
                    current_max_reward=current_max_reward,
                )
                """
                # try:
                #    self.global_params.states_actions_counter[
                #        episode, state, action
                #    ] += 1
                # except KeyError:
                #    self.global_params.states_actions_counter[
                #        episode, state, action
                #    ] = 1

                # print_messages(
                #    "",
                #    next_state=next_state,
                #    state=state,
                # )
                # env.display_manager.render()
                # render params

                render_params_left_bottom(
                    episode=episode,
                    step=step,
                    # action=action,
                    v=action[0][0],  # for continuous actions
                    w=action[0][1],  # for continuous actions
                    # state=state,
                    reward_in_step=reward,
                    cumulated_reward=cumulated_reward,
                    _="------------------------",
                    current_max_reward=current_max_reward,
                    best_epoch=best_epoch,
                    best_step=best_step,
                    done=done,
                    time_per_step=end_step - start_step,
                )

                # best episode and step's stats
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    best_epoch_training_time = (
                        time.time() - start_time_epoch
                    )  # datetime.now() - start_time_epoch
                    self.global_params.actions_rewards["episode"].append(episode)
                    self.global_params.actions_rewards["step"].append(step)
                    self.global_params.actions_rewards["reward"].append(reward)
                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step:
                    save_dataframe_episodes(
                        self.environment.environment,
                        self.global_params.metrics_data_dir,
                        self.global_params.aggr_ep_rewards,
                        self.global_params.actions_rewards,
                    )
                    log.logger.info(
                        f"SHOWING BATCH OF STEPS\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"current epoch = {episode}\n"
                        f"current step = {step}\n"
                        f"best epoch so far = {best_epoch}\n"
                        f"best step so far = {best_step}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )
                    # print_messages(
                    #    "SHOWING BATCH OF STEPS",
                    #    episode=episode,
                    #    step=step,
                    #    cumulated_reward=cumulated_reward,
                    #    best_epoch=best_epoch,
                    #    best_step=best_step,
                    #    current_max_reward=current_max_reward,
                    #    best_epoch_training_time=best_epoch_training_time,
                    # )

                # Reach Finish Line!!!
                if env.is_finish:
                    save_actorcritic_model(
                        ac_agent,
                        self.global_params,
                        self.algoritmhs_params,
                        self.environment.environment,
                        cumulated_reward,
                        episode,
                        "FINISHLINE",
                    )

                    print_messages(
                        "FINISH LINE",
                        episode=episode,
                        step=step,
                        cumulated_reward=cumulated_reward,
                    )
                    log.logger.info(
                        f"\nFINISH LINE\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                    )

                # End epoch
                if step > self.env_params.estimated_steps:
                    done = True
                    save_actorcritic_model(
                        ac_agent,
                        self.global_params,
                        self.algoritmhs_params,
                        self.environment.environment,
                        cumulated_reward,
                        episode,
                        "STEPSCOMPLETED",
                    )

                    print_messages(
                        "STEPS COMPLETED",
                        episode=episode,
                        step=step,
                        cumulated_reward=cumulated_reward,
                    )
                    log.logger.info(
                        f"\nSTEPS COMPLETED\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                    )

                # check out for Carla Server (end of every step)
                ## ----------- checking for Carla Server is working
                env.checking_carla_server(self.environment.environment["town"])

            ########################################
            # collect stats in every epoch
            #
            ########################################

            ############ intrinsic
            finish_time_epoch = time.time()  # datetime.now()

            self.global_params.im_general_ddpg["episode"].append(episode)
            self.global_params.im_general_ddpg["step"].append(step)
            self.global_params.im_general_ddpg["cumulated_reward"].append(
                cumulated_reward
            )
            self.global_params.im_general_ddpg["epoch_time"].append(
                finish_time_epoch - start_time_epoch
            )
            self.global_params.im_general_ddpg["lane_changed"].append(
                len(env.lane_changing_hist)
            )
            self.global_params.im_general_ddpg["distance_to_finish"].append(
                env.dist_to_finish
            )

            # print(f"{self.global_params.time_steps =}")

            ### FPS
            fps_m = [values for values in self.global_params.time_steps.values()]
            fps_mean = sum(fps_m) / len(self.global_params.time_steps)

            sorted_time_steps = OrderedDict(
                sorted(self.global_params.time_steps.items(), key=lambda x: x[1])
            )
            fps_median = median(sorted_time_steps.values())
            # print(f"{fps_mean =}")
            # print(f"{fps_median =}")
            self.global_params.im_general_ddpg["FPS_avg"].append(fps_mean)
            self.global_params.im_general_ddpg["FPS_median"].append(fps_median)

            stats_frame = StatsDataFrame()
            stats_frame.save_dataframe_stats(
                self.environment.environment,
                self.global_params.metrics_graphics_dir,
                self.global_params.im_general_ddpg,
            )

            ########################################
            #
            #
            ########################################

            # Save best lap
            if (
                cumulated_reward - self.environment.environment["rewards"]["penal"]
            ) >= current_max_reward:
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
                save_carla_dataframe_episodes(
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
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"current_max_reward = {current_max_reward}\n"
                    f"steps = {step}\n"
                )

            # end of training by:
            # training time setting: 2 hours, 15 hours...
            # num epochs

            if (
                datetime.now() - timedelta(hours=self.global_params.training_time)
                > start_time
            ) or (episode > self.env_params.total_episodes):
                if (
                    cumulated_reward - self.environment.environment["rewards"]["penal"]
                ) >= current_max_reward:
                    save_actorcritic_model(
                        ac_agent,
                        self.global_params,
                        self.algoritmhs_params,
                        self.environment.environment,
                        cumulated_reward,
                        episode,
                        "FINISHTIME",
                    )
                    log.logger.info(
                        f"\nTraining Time over\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"epoch = {episode}\n"
                        f"step = {step}\n"
                    )

                break

            #####################################################
            # save best values every save_episode times
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

                self.global_params.aggr_ep_rewards = save_batch(
                    episode,
                    step,
                    start_time_epoch,
                    start_time,
                    self.global_params,
                    self.env_params,
                )
                save_carla_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )

                if max_reward > current_max_reward:
                    save_actorcritic_model(
                        ac_agent,
                        self.global_params,
                        self.algoritmhs_params,
                        self.environment.environment,
                        cumulated_reward,
                        episode,
                        "BATCH",
                    )

            ## ------------ destroy actors
            env.destroy_all_actors()
        # ----------- end for

        """
        ### States, Actions counter in every epoch
        for key, value in self.global_params.states_actions_counter.items():
            self.global_params.im_state_actions_counter["epoch"].append(key[0])
            self.global_params.im_state_actions_counter["state"].append(key[1])
            self.global_params.im_state_actions_counter["action"].append(key[2])
            self.global_params.im_state_actions_counter["counter"].append(value)
        stats_frame.save_dataframe_stats(
            self.environment.environment,
            self.global_params.metrics_graphics_dir,
            self.global_params.im_state_actions_counter,
            sac=True,
        )
        """

        env.close()
