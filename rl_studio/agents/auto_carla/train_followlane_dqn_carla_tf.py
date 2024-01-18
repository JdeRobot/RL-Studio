from collections import Counter, OrderedDict, deque
from datetime import datetime, timedelta
import glob
import math

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

# from rl_studio.agents.auto_carla.sources import settings
from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesDDPGCarla,
    LoadEnvVariablesSB3Carla,
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

# from rl_studio.envs.gazebo.gazebo_envs import *
# from rl_studio.envs.carla.carla_env import CarlaEnv
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
import multiprocessing libraries and PythonProgramming libraries
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
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


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


##################################################################################
#
# vars
##################################################################################

correct_normalized_distance = {
    20: -0.07,
    30: -0.1,
    40: -0.13,
    50: -0.17,
    60: -0.2,
    70: -0.23,
    80: -0.26,
    90: -0.3,
    100: -0.33,
    110: -0.36,
    120: -0.4,
    130: -0.42,
    140: -0.46,
    150: -0.49,
    160: -0.52,
    170: -0.56,
    180: -0.59,
    190: -0.62,
    200: -0.65,
    210: -0.69,
    220: -0.72,
}
correct_pixel_distance = {
    20: 343,
    30: 353,
    40: 363,
    50: 374,
    60: 384,
    70: 394,
    80: 404,
    90: 415,
    100: 425,
    110: 436,
    120: 446,
    130: 456,
    140: 467,
    150: 477,
    160: 488,
    170: 498,
    180: 508,
    190: 518,
    200: 528,
    210: 540,
    220: 550,
}


# ==============================================================================
# -- class CarlaEnv -------------------------------------------------------------
# ==============================================================================


class CarlaEnv(gym.Env):
    def __init__(self, car, sensor_camera_rgb, sensor_camera_lanedetector, config):
        super(CarlaEnv, self).__init__()
        ## --------------- init env
        # FollowLaneEnv.__init__(self, **config)
        ## --------------- init class variables
        FollowLaneCarlaConfig.__init__(self, **config)

        # print(f"{config =}")

        self.image_raw_from_topic = None
        self.image_camera = None
        # self.sensor = config["sensor"]

        # Image
        self.image_resizing = config["image_resizing"] / 100
        self.new_image_size = config["new_image_size"]
        self.raw_image = config["raw_image"]
        self.height = int(config["height_image"] * self.image_resizing)
        self.width = int(config["width_image"] * self.image_resizing)
        self.center_image = int(config["center_image"] * self.image_resizing)
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        # self.telemetry_mask = config["telemetry_mask"]
        self.poi = config["x_row"][0]
        self.image_center = None
        self.right_lane_center_image = config["center_image"] + (
            config["center_image"] // 2
        )
        self.lower_limit = config["lower_limit"]

        # States
        self.state_space = config["states"]
        self.states_entry = config["states_entry"]
        if self.state_space == "spn":
            self.x_row = [i for i in range(1, int(self.height / 2) - 1)]
        else:
            self.x_row = config["x_row"]

        # Actions
        self.actions_space = config["action_space"]
        self.actions = config["actions"]

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]

        # Pose
        self.town = config["town"]
        self.random_pose = config["random_pose"]
        self.alternate_pose = config["alternate_pose"]
        self.init_pose_number = config["init_pose_number"]
        self.finish_pose_number = config["finish_pose_number"]
        self.start_alternate_pose = config["start_alternate_pose"]
        self.finish_alternate_pose = config["finish_alternate_pose"]

        # print(f"{self.alternate_pose =}\n")
        # print(f"{self.start_alternate_pose =}\n")

        self.waypoints_meters = config["waypoints_meters"]
        self.waypoints_init = config["waypoints_init"]
        self.waypoints_target = config["waypoints_target"]
        self.waypoints_lane_id = config["waypoints_lane_id"]
        self.waypoints_road_id = config["waypoints_road_id"]
        self.max_target_waypoint_distance = config["max_target_waypoint_distance"]

        ###############################################################
        #
        # gym/stable-baselines3 interface
        ###############################################################

        ######## Actions Gym based
        print(f"{self.state_space = } and {self.actions_space =}")
        # print(f"{len(self.actions) =}")
        # print(f"{type(self.actions) =}")
        # print(f"{self.actions =}")
        # Discrete Actions
        if self.actions_space == "carla_discrete":
            self.action_space = spaces.Discrete(len(self.actions))
        else:  # Continuous Actions
            actions_to_array = np.array(
                [list(self.actions["v"]), list(self.actions["w"])]
            )
            print(f"{actions_to_array =}")
            print(f"{actions_to_array[0] =}")
            print(f"{actions_to_array[1] =}")
            self.action_space = spaces.Box(
                low=actions_to_array[0],
                high=actions_to_array[1],
                shape=(2,),
                dtype=np.float32,
            )
            print(f"{self.action_space.low = }")
            print(f"{self.action_space.high = }")
            print(f"{self.action_space.low[0] = }")
            print(f"{self.action_space.high[0] = }")

        print(f"{self.action_space =}")

        ######## observations Gym based
        # image
        if self.state_space == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )
        else:  # Discrete observations vector = [0.0, 3.6, 8.9]
            # TODO: change x_row for other list
            self.observation_space = spaces.Discrete(len(self.x_row))  # temporary

        print(f"{self.observation_space = }")

        #########################################################################

        # self.params = config
        # print(f"{self.params =}\n")

        self.car = car
        self.sensor_camera_rgb = sensor_camera_rgb
        self.sensor_camera_lanedetector = sensor_camera_lanedetector
        self.params = {}
        self.is_finish = None
        self.dist_to_finish = None
        self.collision_hist = []

        self.right_line_in_pixels = None
        self.ground_truth_pixel_values = None
        self.dist_normalized = None
        self.states = None
        self.state_right_lines = None
        self.drawing_lines_states = []
        self.drawing_numbers_states = []

        self.lane_changing_hist = []

    #####################################################################################
    #
    #                                       RESET
    #
    #####################################################################################
    def reset(self, seed=None, options=None):
        """
        state = vector / simplified perception
        actions = discrete
        """

        ############################################################################
        ########### --- calculating STATES

        ##### EATA PARTE NOS SIRVE PARA ENCONTRAR EL CENTRO SOLO CON LA LINEA DERECHA
        while self.sensor_camera_rgb.front_rgb_camera is None:
            print(f"RESET() ----> {self.sensor_camera_rgb.front_rgb_camera = }")
            time.sleep(0.2)

        # mask = self.preprocess_image(self.front_red_mask_camera)
        # mask = self.preprocess_image(self.sensor_camera.front_red_mask_camera)
        # self.right_line_in_pixels = self.calculate_right_line(mask, self.x_row)

        """
        ##### EATA PARTE NOS SIRVE PARA ENCONTRAR EL CENTRO SOLO CON EL LANE DETECTOR
        (
            image_rgb_lanedetector,
            left_mask,
            right_mask,
        ) = self.sensor_camera_lanedetector.detect(
            self.sensor_camera_rgb.front_rgb_camera
        )
        mask = self.preprocess_image_lane_detector(image_rgb_lanedetector)
        lane_centers_in_pixels, _, _ = self.calculate_lane_centers_with_lane_detector(
            mask
        )

        pixels_in_state = mask.shape[1] / self.num_regions
        self.states = [
            int(value / pixels_in_state)
            for _, value in enumerate(lane_centers_in_pixels)
        ]
        
        """
        self.states = [self.num_regions // 2 for i, _ in enumerate(self.x_row)]
        print(f"\n\t{self.states =}")
        # states_size = len(self.states)

        """
        AutoCarlaUtils.show_image(
            "mask",
            mask,
            50,
            500,
        )

        AutoCarlaUtils.show_image(
            "LaneDetector RGB",
            image_rgb_lanedetector[(image_rgb_lanedetector.shape[0] // 2) :],
            600,
            500,
        )

        AutoCarlaUtils.show_image(
            "front RGB",
            self.sensor_camera_rgb.front_rgb_camera[
                (self.sensor_camera_rgb.front_rgb_camera.shape[0] // 2) :
            ],
            1250,
            500,
        )
        """

        # print(f"{states =} and {states_size =}")
        return self.states  # , states_size

    #####################################################################################
    #
    #                                       STEP
    #
    #####################################################################################
    def step(self, action):
        """
        state: sp
        actions: continuous
        only right line
        """

        self.control_discrete_actions(action)
        print(f"\n\tSTEP() POR AQUI")

        ########### --- calculating STATES
        mask = self.preprocess_image(self.sensor_camera_red_mask.front_red_mask_camera)

        ########### --- Calculating center ONLY with right line
        self.right_line_in_pixels = self.calculate_right_line(mask, self.x_row)

        image_center = mask.shape[1] // 2
        # dist = [
        #    image_center - right_line_in_pixels[i]
        #    for i, _ in enumerate(right_line_in_pixels)
        # ]
        self.dist_normalized = [
            float((image_center - self.right_line_in_pixels[i]) / image_center)
            for i, _ in enumerate(self.right_line_in_pixels)
        ]

        ## STATES

        # pixels_in_state = mask.shape[1] // self.num_regions
        # self.state_right_lines = [
        #    i for i in range(1, mask.shape[1]) if i % pixels_in_state == 0
        # ]
        # self.states = [
        #    int(value / pixels_in_state)
        #    for _, value in enumerate(self.right_line_in_pixels)
        # ]

        # non regular states: 1 to n-2 in center. n-1 right, n left
        size_lateral_states = 140
        size_center_states = mask.shape[1] - (size_lateral_states * 2)
        pixel_center_states = int(size_center_states / (self.num_regions - 2))

        self.states = [
            int(((value - size_lateral_states) / pixel_center_states) + 1)
            if (mask.shape[1] - size_lateral_states) > value > size_lateral_states
            else self.num_regions - 1
            if value >= (mask.shape[1] - size_lateral_states)
            else self.num_regions
            for _, value in enumerate(self.right_line_in_pixels)
        ]

        # drawing lines and numbers states in image
        self.drawing_lines_states = [
            size_lateral_states + (i * pixel_center_states)
            for i in range(1, self.num_regions - 1)
        ]
        # self.drawing_lines_states.append(size_lateral_states)
        self.drawing_lines_states.insert(0, size_lateral_states)

        self.drawing_numbers_states = [
            i if i > 0 else self.num_regions for i in range(0, self.num_regions)
        ]
        # print(f"\n{self.drawing_lines_states}")
        # print(f"\n{self.drawing_numbers_states}")

        ## -------- Ending Step()...
        done = False
        ground_truth_normal_values = [
            correct_normalized_distance[value] for i, value in enumerate(self.x_row)
        ]
        # reward, done = self.autocarlarewards.rewards_right_line(
        #    dist_normalized, ground_truth_normal_values, self.params
        # )
        reward, done = self.autocarlarewards.rewards_sigmoid_only_right_line(
            self.dist_normalized, ground_truth_normal_values
        )

        ## -------- ... or Finish by...
        if len(self.collision_hist) > 0:  # crashed you, baby
            done = True
            # reward = -100
            print(f"crash")

        self.is_finish, self.dist_to_finish = AutoCarlaUtils.finish_fix_number_target(
            self.params["location"],
            self.finish_alternate_pose,
            self.finish_pose_number,
            self.max_target_waypoint_distance,
        )
        if self.is_finish:
            print(f"Finish!!!!")
            done = True

        self.ground_truth_pixel_values = [
            correct_pixel_distance[value] for i, value in enumerate(self.x_row)
        ]

        return self.states, reward, done, {}

    ##################################################
    #
    #   Control
    ###################################################

    def control_discrete_actions(self, action):
        """
        working with LaneDetector
        """
        t = self.car.car.get_transform()
        v = self.car.car.get_velocity()  # returns in m/sec
        c = self.car.car.get_control()
        w = self.car.car.get_angular_velocity()  # returns in deg/sec
        a = self.car.car.get_acceleration()
        self.params["speed"] = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.params["steering_angle"] = w
        # self.params["Steering_angle"] = steering_angle
        self.params["Steer"] = c.steer
        self.params["location"] = (t.location.x, t.location.y)
        self.params["Throttle"] = c.throttle
        self.params["Brake"] = c.brake
        self.params["height"] = t.location.z
        self.params["Acceleration"] = math.sqrt(a.x**2 + a.y**2 + a.z**2)

        ## Applied throttle, brake and steer
        curr_speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        throttle = self.actions[action][0]
        steer = self.actions[action][1]
        # print(f"in STEP() {throttle = } and {steer = }")

        target_speed = 30
        brake = 0

        if curr_speed > target_speed:
            throttle = 0.45

        self.car.car.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

    #####################################################################################
    # ---   methods
    #####################################################################################

    def preprocess_image(self, img):
        """
        image is trimming from top to middle
        """
        ## first, we cut the upper image
        height = img.shape[0]
        image_middle_line = (height) // 2
        img_sliced = img[image_middle_line:]
        ## calculating new image measurements
        # height = img_sliced.shape[0]
        ## -- convert to GRAY
        gray_mask = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        ## --  apply mask to convert in Black and White
        theshold = 50
        # _, white_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)
        _, white_mask = cv2.threshold(gray_mask, theshold, 255, cv2.THRESH_BINARY)

        return white_mask

    def calculate_right_line(self, mask, x_row):
        """
        calculates distance from center to right line
        This distance will be using as a error from center lane
        """
        ## get total lines in every line point
        lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]

        ### ----------------- from right to left
        lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        # print(f"{lines_inversed = }")
        inv_index_right = [
            np.argmax(lines_inversed[x]) for x, _ in enumerate(lines_inversed)
        ]

        index_right = [
            mask.shape[1] - inv_index_right[x] if inv_index_right[x] != 0 else 0
            for x, _ in enumerate(inv_index_right)
        ]

        return index_right

    def preprocess_image_lane_detector(self, image):
        """
        image from lane detector
        """
        ## first, we cut the upper image
        height = image.shape[0]
        image_middle_line = (height) // 2
        img_sliced = image[image_middle_line:]

        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(img_sliced, kernel, iterations=1)
        hsv = cv2.cvtColor(img_erosion, cv2.COLOR_RGB2HSV)
        gray_hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
        _, gray_mask = cv2.threshold(gray_hsv, 200, 255, cv2.THRESH_BINARY)

        return gray_mask

    def calculate_lane_centers_with_lane_detector(self, mask):
        """
        using Lane Detector model for calculating the center
        """
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        index_left = [np.argmax(lines[x]) for x, _ in enumerate(lines)]
        left_offset, right_offset = 20, 10
        index_left_plus_offset = [
            index_left[x] + left_offset if index_left[x] != 0 else 0
            for x, _ in enumerate(index_left)
        ]
        second_lines_from_left_right = [
            lines[x][index_left_plus_offset[x] :] for x, _ in enumerate(lines)
        ]
        ## ---------------- calculating index in second line
        try:
            index_right_no_offset = [
                np.argmax(second_lines_from_left_right[x])
                for x, _ in enumerate(second_lines_from_left_right)
            ]
        except:
            secod_lines_from_left_right = [value for _, value in enumerate(self.x_row)]
            index_right_no_offset = [
                np.argmax(second_lines_from_left_right[x])
                for x, _ in enumerate(second_lines_from_left_right)
            ]

        index_right_plus_offsets = [
            index_right_no_offset[x] + left_offset + right_offset
            for x, _ in enumerate(index_right_no_offset)
        ]

        index_right = [
            right + left for right, left in zip(index_right_plus_offsets, index_left)
        ]

        centers = [
            ((right - left) // 2) + left for right, left in zip(index_right, index_left)
        ]

        return centers, index_left, index_right


# ==============================================================================
# -- class RGB CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraRGBSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.front_rgb_camera = None

        self.world = self._parent.get_world()
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", f"640")
        bp.set_attribute("image_size_y", f"480")
        bp.set_attribute("fov", f"110")

        self.sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            attach_to=self._parent,
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraRGBSensor._rgb_image(weak_self, image))

    @staticmethod
    def _rgb_image(weak_self, image):
        """weakref"""
        self = weak_self()
        if not self:
            return
        image = np.array(image.raw_data)
        image = image.reshape((480, 640, 4))
        # image = image.reshape((512, 1024, 4))
        image = image[:, :, :3]
        # self._data_dict["image"] = image3
        self.front_rgb_camera = image


# ==============================================================================
# -- Red Mask CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraRedMaskSemanticSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.front_red_mask_camera = None

        self.world = self._parent.get_world()
        bp = self.world.get_blueprint_library().find(
            "sensor.camera.semantic_segmentation"
        )
        bp.set_attribute("image_size_x", f"640")
        bp.set_attribute("image_size_y", f"480")
        bp.set_attribute("fov", f"110")

        self.sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            attach_to=self._parent,
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraRedMaskSemanticSensor._red_mask_semantic_image_callback(
                weak_self, image
            )
        )

    @staticmethod
    def _red_mask_semantic_image_callback(weak_self, image):
        """weakref"""
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        hsv_nemo = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

        if (
            self.world.get_map().name == "Carla/Maps/Town07"
            or self.world.get_map().name == "Carla/Maps/Town04"
            or self.world.get_map().name == "Carla/Maps/Town07_Opt"
            or self.world.get_map().name == "Carla/Maps/Town04_Opt"
        ):
            light_sidewalk = (42, 200, 233)
            dark_sidewalk = (44, 202, 235)
        else:
            light_sidewalk = (151, 217, 243)
            dark_sidewalk = (153, 219, 245)

        light_pavement = (149, 127, 127)
        dark_pavement = (151, 129, 129)

        mask_sidewalk = cv2.inRange(hsv_nemo, light_sidewalk, dark_sidewalk)
        # result_sidewalk = cv2.bitwise_and(array, array, mask=mask_sidewalk)

        mask_pavement = cv2.inRange(hsv_nemo, light_pavement, dark_pavement)
        # result_pavement = cv2.bitwise_and(array, array, mask=mask_pavement)

        # Adjust according to your adjacency requirement.
        kernel = np.ones((3, 3), dtype=np.uint8)

        # Dilating masks to expand boundary.
        color1_mask = cv2.dilate(mask_sidewalk, kernel, iterations=1)
        color2_mask = cv2.dilate(mask_pavement, kernel, iterations=1)

        # Required points now will have both color's mask val as 255.
        common = cv2.bitwise_and(color1_mask, color2_mask)
        SOME_THRESHOLD = 0

        # Common is binary np.uint8 image, min = 0, max = 255.
        # SOME_THRESHOLD can be anything within the above range. (not needed though)
        # Extract/Use it in whatever way you want it.
        intersection_points = np.where(common > SOME_THRESHOLD)

        # Say you want these points in a list form, then you can do this.
        pts_list = [[r, c] for r, c in zip(*intersection_points)]
        # print(pts_list)

        # for x, y in pts_list:
        #    image_2[x][y] = (255, 0, 0)

        # red_line_mask = np.zeros((400, 500, 3), dtype=np.uint8)
        red_line_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

        for x, y in pts_list:
            red_line_mask[x][y] = (255, 0, 0)

        # t_end = self.timer.time()
        # self.time_processing += t_end - t_start
        # self.tics_processing += 1

        red_line_mask = cv2.cvtColor(red_line_mask, cv2.COLOR_BGR2RGB)
        self.front_red_mask_camera = red_line_mask

        # AutoCarlaUtils.show_image(
        #    "states",
        #    self.front_red_mask_camera,
        #    600,
        #    400,
        # )
        # if self.front_red_mask_camera is not None:
        #    time.sleep(0.01)
        #    print(f"self.front_red_mask_camera leyendo")
        #    print(f"in _red_mask_semantic_image_callback() {time.time()}")


# ==============================================================================
# -- class LaneDetector -------------------------------------------------------------
# ==============================================================================
class LaneDetector:
    def __init__(self, model_path: str):
        torch.cuda.empty_cache()
        self.__model: torch.nn.Module = torch.load(model_path)
        self.__model.eval()

    def detect(self, img_array: np.array) -> tuple:
        with torch.no_grad():
            image_tensor = img_array.transpose(2, 0, 1).astype("float32") / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            back, left, right = (
                torch.softmax(self.__model.forward(x_tensor), dim=1).cpu().numpy()[0]
            )

        res, left_mask, right_mask = self.lane_detection_overlay(img_array, left, right)

        return res, left_mask, right_mask

    def lane_detection_overlay(
        self, image: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray
    ) -> tuple:
        """
        @type left_mask: object
        """
        res = np.copy(image)

        cv2.erode(left_mask, (7, 7), 4)
        cv2.dilate(left_mask, (7, 7), 4)

        cv2.erode(right_mask, (7, 7), 4)
        cv2.dilate(right_mask, (7, 7), 4)

        left_mask = self.image_polyfit(left_mask)
        right_mask = self.image_polyfit(right_mask)

        # We show only points with probability higher than 0.07
        res[left_mask > 0.1, :] = [255, 0, 0]
        res[right_mask > 0.1, :] = [0, 0, 255]

        return res, left_mask, right_mask

    def image_polyfit(self, image: np.ndarray) -> np.ndarray:
        img = np.copy(image)
        img[image > 0.1] = 255

        indices = np.where(img == 255)

        if len(indices[0]) == 0:
            return img
        grade = 1
        coefficients = np.polyfit(indices[0], indices[1], grade)

        x = np.linspace(0, img.shape[1], num=2500)
        y = np.polyval(coefficients, x)
        points = np.column_stack((x, y)).astype(int)

        valid_points = []

        for point in points:
            # if (0 < point[1] < 1023) and (0 < point[0] < 509):
            if (0 < point[1] < image.shape[1]) and (0 < point[0] < image.shape[0]):
                valid_points.append(point)

        valid_points = np.array(valid_points)
        polyfitted = np.zeros_like(img)
        polyfitted[tuple(valid_points.T)] = 255

        return polyfitted


# ==============================================================================
# -- New Car -------------------------------------------------------------
# ==============================================================================
class NewCar(object):
    def __init__(self, parent_actor, init_positions, init=None):
        self.car = None
        self._parent = parent_actor
        self.world = self._parent
        vehicle = self.world.get_blueprint_library().filter("vehicle.*")[0]

        if init is None:
            pose_init = np.random.randint(0, high=len(init_positions))
        else:
            pose_init = init

        location = carla.Transform(
            carla.Location(
                x=init_positions[pose_init][0],
                y=init_positions[pose_init][1],
                z=init_positions[pose_init][2],
            ),
            carla.Rotation(
                pitch=init_positions[pose_init][3],
                yaw=init_positions[pose_init][4],
                roll=init_positions[pose_init][5],
            ),
        )

        self.car = self.world.spawn_actor(vehicle, location)
        while self.car is None:
            self.car = self.world.spawn_actor(vehicle, location)


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################


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
        # print(f"{config =}\n")

        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesSB3Carla(config)
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

        log = LoggingHandler(self.log_file)
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

        dqn_agent = DQN(
            self.environment.environment,
            self.algoritmhs_params,
            len(self.global_params.actions_set),
            len(self.environment.environment["x_row"]),
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
            print(f"\n maps in carla 0.9.13: {self.client.get_available_maps()}\n")

            self.world = self.client.load_world(self.environment.environment["town"])

            # Sync mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1  # read: https://carla.readthedocs.io/en/0.9.13/adv_synchrony_timestep/# Phisics substepping
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

            ####################################################################################
            # FOR
            #
            ####################################################################################

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

                ## LaneDetector Camera ---------------
                self.sensor_camera_lanedetector = LaneDetector(
                    "models/fastai_torch_lane_detector_model.pth"
                )
                self.actor_list.append(self.sensor_camera_lanedetector)

                # print(
                #    f"in main() {self.sensor_camera_rgb.front_rgb_camera} in {time.time()}"
                # )
                # print(
                #    f"in main() {self.sensor_camera_red_mask.front_red_mask_camera} in {time.time()}"
                # )

                ############################################################################
                # ENV, RESET, DQN-AGENT, FOR
                ############################################################################

                env = CarlaEnv(
                    self.new_car,
                    self.sensor_camera_rgb,  # img to process
                    # self.sensor_camera_red_mask, # sensor to process image
                    self.sensor_camera_lanedetector,  # sensor to process image
                    self.environment.environment,
                )
                # env = gym.make(self.env_params.env_name, **self.environment.environment)
                # check_env(env, warn=True)

                self.world.tick()

                ############################
                state = env.reset()
                # print(f"{state = } and {state_size = }")

                # self.world.tick()

                ######################################################################################
                #
                # STEPS
                ######################################################################################

                while not done:
                    # print(f"\n{episode =} , {step = }")
                    # print(f"{state = } and {state_size = }")

                    self.world.tick()

                    if np.random.random() > epsilon:
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
                    end_step = time.time()

                    # Every step we update replay memory and train main network
                    # agent_dqn.update_replay_memory((state, action, reward, nextState, done))
                    dqn_agent.update_replay_memory(
                        (state, action, reward, new_state, done)
                    )
                    dqn_agent.train(done, step)

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
                        observation=state,
                        new_observation=new_state,
                        action=action,
                        throttle=self.global_params.actions_set[action][
                            0
                        ],  # this case for discrete
                        steer=self.global_params.actions_set[action][
                            1
                        ],  # this case for discrete
                        v_km_h=env.params["speed"],
                        w_deg_sec=env.params["steering_angle"],
                        epsilon=epsilon,
                        reward_in_step=reward,
                        cumulated_reward_in_this_episode=cumulated_reward,
                        _="------------------------",
                        best_episode_until_now=best_epoch,
                        in_best_step=best_step,
                        with_highest_reward=int(current_max_reward),
                        # in_best_epoch_training_time=best_epoch_training_time,
                    )

                    AutoCarlaUtils.show_image(
                        "RGB",
                        self.sensor_camera_rgb.front_rgb_camera,
                        50,
                        50,
                    )
                    AutoCarlaUtils.show_image(
                        "segmentation_cam",
                        self.sensor_camera_red_mask.front_red_mask_camera,
                        500,
                        50,
                    )

                    AutoCarlaUtils.show_image_only_right_line(
                        "front RGB",
                        # self.front_rgb_camera[(self.front_rgb_camera.shape[0] // 2) :],
                        self.sensor_camera_rgb.front_rgb_camera[
                            (self.sensor_camera_rgb.front_rgb_camera.shape[0] // 2) :
                        ],
                        1,
                        env.right_line_in_pixels,
                        env.ground_truth_pixel_values,
                        env.dist_normalized,
                        new_state,
                        env.x_row,
                        1250,
                        10,
                        env.drawing_lines_states,
                        env.drawing_numbers_states,
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
