from collections import Counter, OrderedDict, deque
from datetime import datetime, timedelta
import glob
import math
import resource

# from multiprocessing import Process, cpu_count, Queue, Value, Array
# from threading import Thread
# import subprocess
from statistics import median
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import time
import weakref

import carla
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import ray
from reloading import reloading
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.policies import MlpPolicy
import tensorflow as tf
import torch

# import keras.backend.tensorflow_backend as backend
from tqdm import tqdm
import sys

from rl_studio.agents.auto_carla.actors_sensors import (
    NewCar,
    CameraRGBSensor,
    CameraRedMaskSemanticSensor,
    # LaneDetector,
)
from rl_studio.agents.auto_carla.carla_env import CarlaEnv

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadGlobalParams,
    LoadEnvVariablesDQNCarla,
)
from rl_studio.agents.auto_carla.utils import (
    LoggingHandler,
    Logger,
    LoggerAllInOne,
)
from rl_studio.agents.utils import (
    render_params,
    render_params_left_bottom,
    save_dataframe_episodes,
    save_carla_dataframe_episodes,
    save_batch,
    save_best_episode,
    # LoggingHandler,
    print_messages,
)
from rl_studio.agents.utilities.plot_stats import MetricsPlot, StatsDataFrame
from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
    OUActionNoise,
    Buffer,
    DDPGAgent,
)
from rl_studio.algorithms.dqn_keras import (
    ModifiedTensorBoard,
    DQN,
)
from rl_studio.algorithms.dqn_keras_parallel import (
    ModifiedTensorBoard,
    DQNMultiprocessing,
)
from rl_studio.algorithms.utils import (
    save_actorcritic_model,
)
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig
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

##################################################################
"""
import multiprocessing libraries
"""
from multiprocessing import Process, cpu_count, Queue, Value, Array
from threading import Thread
import subprocess

import rl_studio.agents.auto_carla.sources.settings
from rl_studio.agents.auto_carla.sources.cage import check_weights_size


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


##################################################################################
#
# GPUs management
##################################################################################

tf.debugging.set_log_device_placement(False)

# Sharing GPU
gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # auto memory configuration

logical_gpus = tf.config.list_logical_devices("GPU")
print(
    f"\n\tIn train_followlane_dqn_carla_tf.py ---> {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs"
)

############ 1 phisical GPU + 2 logial GPUs
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     # Create 2 virtual GPUs with 1GB memory each
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [
#                 tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                 tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                 tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                 tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#             ],
#         )
#         logical_gpus = tf.config.list_logical_devices("GPU")
#         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


# Init Ray
# ray.init(ignore_reinit_error=True)


#############################################################################################
#
# Trainer
#############################################################################################


class TrainerFollowLaneDQNAutoCarlaTF:
    """

    Mode: training
    Task: Follow Lane
    Algorithm: DQN
    Agent: Auto
    Simulator: Carla
    Framework: TF
    Weather: Static
    Traffic: No

    The most simplest environment
    """

    def __init__(self, config):

        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesDQNCarla(config)
        self.global_params = LoadGlobalParams(config)

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.recorders_carla_dir}", exist_ok=True)

        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"

        # CarlaEnv.__init__(self)

        self.world = None
        self.client = None
        self.front_rgb_camera = None
        self.front_red_mask_camera = None
        self.front_lanedetector_camera = None
        self.actor_list = []

    #########################################################################
    # Main
    #########################################################################

    def main(self):
        """ """
        #########################################################################
        # Vars
        #########################################################################

        # log = LoggingHandler(self.log_file)
        # log = LoggerAllInOne(self.log_file)
        log = Logger(self.log_file)
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
        epsilon_decay = epsilon / (self.env_params.total_episodes // 2)

        ### DQN declaration
        ### DQN size: [centers_n, line_borders_n * 2, v, w, angle]
        ### DQN size = (x_row * 3) + 3
        DQN_size = (len(self.environment.environment["x_row"]) * 3) + 3

        dqn_agent = DQN(
            self.environment.environment,
            self.algoritmhs_params,
            len(self.global_params.actions_set),
            # len(self.environment.environment["x_row"]),
            DQN_size,
            self.global_params.models_dir,
            self.global_params,
        )
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        try:
            #########################################################################
            # Vars
            #########################################################################
            # print(f"\n al inicio de Main() {self.environment.environment =}\n")
            self.client = carla.Client(
                self.environment.environment["carla_server"],
                self.environment.environment["carla_client"],
            )
            self.client.set_timeout(3.0)
            print(
                f"\n In TrainerFollowLaneDQNAutoCarlaTF/main() ---> maps in carla 0.9.13: {self.client.get_available_maps()}\n"
            )

            self.world = self.client.load_world(self.environment.environment["town"])

            # Sync mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1  # read: https://carla.readthedocs.io/en/0.9.13/adv_synchrony_timestep/# Phisics substepping
            # With 0.05 value, the simulator will take twenty steps (1/0.05) to recreate one second of the simulated world
            self.world.apply_settings(settings)

            # Set up the traffic manager
            traffic_manager = self.client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(0)  # define TM seed for determinism

            self.client.reload_world(False)

            # print(f"{settings.synchronous_mode =}")
            # self.world.tick()
            # print(f"{self.world.tick()=}")

            # Town07 take layers off
            if self.environment.environment["town"] == "Town07_Opt":
                self.world.unload_map_layer(carla.MapLayer.Buildings)
                self.world.unload_map_layer(carla.MapLayer.Decals)
                self.world.unload_map_layer(carla.MapLayer.Foliage)
                self.world.unload_map_layer(carla.MapLayer.Particles)
                self.world.unload_map_layer(carla.MapLayer.Props)

            ## LaneDetector Camera ---------------
            # self.sensor_camera_lanedetector = LaneDetector(
            #    "models/fastai_torch_lane_detector_model.pth"
            # )

            ####################################################################################
            # FOR
            #
            ####################################################################################
            memory_use_in_every_epoch = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss / (1024 * 1024)
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

                self.world.tick()

                self.new_car = NewCar(
                    self.world,
                    self.environment.environment["start_alternate_pose"],
                    self.environment.environment["alternate_pose"],
                )
                # self.actor_list.append(self.new_car.car)

                ############################################
                ## --------------- Sensors ---------------
                ############################################

                ## RGB camera ---------------
                self.sensor_camera_rgb = CameraRGBSensor(self.new_car.car)
                self.actor_list.append(self.sensor_camera_rgb.sensor)

                ## RedMask Camera ---------------
                self.sensor_camera_red_mask = CameraRedMaskSemanticSensor(
                    self.new_car.car
                )
                self.actor_list.append(self.sensor_camera_red_mask.sensor)

                ############################################################################
                # ENV, RESET, DQN-AGENT, FOR
                ############################################################################

                env = CarlaEnv(
                    self.new_car,
                    self.sensor_camera_rgb,  # img to process
                    self.sensor_camera_red_mask,  # sensor to process image
                    self.environment.environment,
                )
                self.world.tick()

                state = env.reset()
                # print(f"\n\tin Training For loop -------> {state =}")
                ######################################################################################
                #
                # STEPS
                ######################################################################################
                memory_use_in_every_step = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss / (1024 * 1024)
                while not done:
                    # print(f"\n{episode =} , {step = }")
                    # print(f"{state = } and {state_size = }")

                    self.world.tick()

                    # epsilon = 0.1  ###### PARA PROBAR-----------OJO QUITARLO
                    if np.random.random() > epsilon:
                        # print(f"\n\tin Training For loop -----> {epsilon =}")
                        action = np.argmax(dqn_agent.get_qs(state))
                    else:
                        # Get random action
                        action = np.random.randint(
                            0, len(self.global_params.actions_set)
                        )

                    # print(f"{action = }\n")

                    ############################
                    start_step = time.time()
                    new_state, reward, done, _ = env.step(action)
                    # print(f"{new_state = } and {reward = } and {done =}")
                    # print(
                    #    f"\n\tin Training For loop ---> {state =}, {action =}, {new_state =}, {reward =}, {done =}"
                    # )

                    # Every step we update replay memory and train main network
                    # agent_dqn.update_replay_memory((state, action, reward, nextState, done))
                    dqn_agent.update_replay_memory(
                        (state, action, reward, new_state, done)
                    )
                    dqn_agent.train(done, step)
                    end_step = time.time()

                    self.world.tick()

                    self.global_params.time_steps[step] = end_step - start_step
                    cumulated_reward += reward
                    state = new_state
                    step += 1

                    ########################
                    ### solo para chequear que funcione
                    #########################

                    # if self.sensor_camera_rgb.front_rgb_camera is not None:
                    #    print(
                    #        f"Again in main() self.sensor_camera_rgb.front_rgb_camera in {time.time()}"
                    #    )
                    # if self.sensor_camera_red_mask.front_red_mask_camera is not None:
                    #    print(
                    #        f"Again in main() self.sensor_camera_red_mask.front_red_mask_camera in {time.time()}"
                    #    )
                    render_params_left_bottom(
                        episode=episode,
                        step=step,
                        epsilon=epsilon,
                        # observation=state,
                        # new_observation=new_state,
                        action=action,
                        throttle=self.global_params.actions_set[action][
                            0
                        ],  # this case for discrete
                        steer=self.global_params.actions_set[action][
                            1
                        ],  # this case for discrete
                        v_km_h=env.params["current_speed"],
                        w_deg_sec=env.params["current_steering_angle"],
                        angle=env.angle,
                        FPS=1 / (end_step - start_step),
                        centers=sum(env.centers_normal) / len(env.centers_normal),
                        reward_in_step=reward,
                        cumulated_reward_in_this_episode=cumulated_reward,
                        memory_use_in_every_epoch=memory_use_in_every_epoch,
                        memory_use_in_every_step=memory_use_in_every_step,
                        _="------------------------",
                        best_episode_until_now=best_epoch,
                        in_best_step=best_step,
                        with_highest_reward=int(current_max_reward),
                        # in_best_epoch_training_time=best_epoch_training_time,
                    )

                    log._warning(
                        f"\n\tepisode = {episode}\n"
                        f"step = {step}\n"
                        f"epsilon = {epsilon}\n"
                        f"state = {state}\n"
                        f"action = {action}\n"
                        f"angle = {env.angle}\n"
                        f"FPS = {1/(end_step - start_step)}\n"
                        f"reward in step = {reward}\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"cumulated_reward_in_this_episode = {cumulated_reward}\n"
                        f"done = {done}\n"
                        f"memory_use_in_every_epoch = {memory_use_in_every_epoch}\n"
                        f"memory_use_in_every_step = {memory_use_in_every_step}\n"
                    )
                    # best episode
                    if current_max_reward <= cumulated_reward and episode > 1:
                        current_max_reward = cumulated_reward
                        best_epoch = episode
                        best_step = step
                        best_epoch_training_time = time.time() - start_time_epoch
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

                    # Reach Finish Line!!!
                    if env.is_finish:
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.model",
                        )
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.h5",
                        )

                        print_messages(
                            "FINISH LINE",
                            episode=episode,
                            step=step,
                            cumulated_reward=cumulated_reward,
                        )

                    ### save in case of completed steps in one episode
                    if step >= self.env_params.estimated_steps:
                        done = True
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.model",
                        )
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.h5",
                        )

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

                #### save best lap in episode
                if (
                    cumulated_reward - self.environment.environment["rewards"]["penal"]
                ) >= current_max_reward and episode > 1:
                    self.global_params.best_current_epoch["best_epoch"].append(
                        best_epoch
                    )
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
                    dqn_agent.model.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BESTLAP_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.model",
                    )
                    dqn_agent.model.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BESTLAP_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.h5",
                    )

                # end episode in time settings: 2 hours, 15 hours...
                # or epochs over
                if (
                    datetime.now() - timedelta(hours=self.global_params.training_time)
                    > start_time
                ) or (episode > self.env_params.total_episodes):
                    log.logger.info(
                        f"\nTraining Time over or num epochs reached\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epoch = {episode}\n"
                        f"step = {step}\n"
                        f"epsilon = {epsilon}\n"
                    )
                    if (
                        cumulated_reward
                        - self.environment.environment["rewards"]["penal"]
                    ) >= current_max_reward:
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_LAPCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.model",
                        )
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_LAPCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.h5",
                        )
                    break

                ### save every save_episode times
                self.global_params.ep_rewards.append(cumulated_reward)
                if not episode % self.env_params.save_episodes:

                    # input("parada en Training al saving el batch.....verificar valores")
                    average_reward = sum(
                        self.global_params.ep_rewards[-self.env_params.save_episodes :]
                    ) / len(
                        self.global_params.ep_rewards[-self.env_params.save_episodes :]
                    )
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
                        epsilon=epsilon,
                    )

                    self.global_params.aggr_ep_rewards["episode"].append(episode)
                    self.global_params.aggr_ep_rewards["step"].append(step)
                    self.global_params.aggr_ep_rewards["avg"].append(average_reward)
                    self.global_params.aggr_ep_rewards["max"].append(max_reward)
                    self.global_params.aggr_ep_rewards["min"].append(min_reward)
                    self.global_params.aggr_ep_rewards["epoch_training_time"].append(
                        (time.time() - start_time_epoch)
                    )
                    self.global_params.aggr_ep_rewards["total_training_time"].append(
                        (datetime.now() - start_time).total_seconds()
                    )
                    if max_reward > current_max_reward:
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BATCH_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.model",
                        )
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BATCH_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}.h5",
                        )
                        save_dataframe_episodes(
                            self.environment.environment,
                            self.global_params.metrics_data_dir,
                            self.global_params.aggr_ep_rewards,
                        )

                #######################################################
                #
                # End render
                #######################################################

                ### reducing epsilon
                if epsilon > epsilon_min:
                    # epsilon *= epsilon_discount
                    epsilon -= epsilon_decay

                # if episode < 4:
                # self.client.apply_batch(
                #    [carla.command.DestroyActor(x) for x in self.actor_list[::-1]]
                # )
                # self.actor_list = []

                for sensor in self.actor_list:
                    if sensor.is_listening:
                        # print(f"is_listening {sensor =}")
                        sensor.stop()

                    # print(f"destroying {sensor =}")
                    sensor.destroy()

                self.new_car.car.destroy()
                self.actor_list = []

            ### save last episode, not neccesarily the best one
            save_dataframe_episodes(
                self.environment.environment,
                self.global_params.metrics_data_dir,
                self.global_params.aggr_ep_rewards,
            )

        ############################################################################
        #
        # finally
        ############################################################################

        finally:
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                traffic_manager.set_synchronous_mode(True)
                print(f"ending training...bye!!")

            # destroy_all_actors()
            # for actor in self.actor_list[::-1]:

            # print(f"{self.actor_list =}")
            # if len(self.actor_list)
            # for actor in self.actor_list:
            #    actor.destroy()

            env.close()
