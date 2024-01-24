from collections import Counter, OrderedDict
from datetime import datetime, timedelta
import glob
import math
from statistics import median
import os
import random
import time
import weakref

import carla
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from reloading import reloading
import tensorflow as tf
from tqdm import tqdm
from stable_baselines3.common.env_checker import check_env
import sys

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
from rl_studio.algorithms.utils import (
    save_actorcritic_model,
)

# from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.carla_env import CarlaEnv
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


# Sharing GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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


class CarlaEnv(gym.Env):
    def __init__(self, car, sensor_camera_red_mask, config):
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
        self.sensor_camera_red_mask = sensor_camera_red_mask
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
        while self.sensor_camera_red_mask.front_red_mask_camera is None:
            # print(
            #    f"RESET() ----> {self.sensor_camera_red_mask.front_red_mask_camera = }"
            # )
            time.sleep(1)

        # mask = self.preprocess_image(self.front_red_mask_camera)
        mask = self.preprocess_image(self.sensor_camera_red_mask.front_red_mask_camera)
        self.right_line_in_pixels = self.calculate_right_line(mask, self.x_row)

        pixels_in_state = mask.shape[1] / self.num_regions
        self.states = [
            int(value / pixels_in_state)
            for _, value in enumerate(self.right_line_in_pixels)
        ]
        # states = [5, 5, 5, 5]
        states_size = len(self.states)

        # print(f"{states =} and {states_size =}")
        return self.states, states_size

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

        self.control_discrete_actions_only_right(action)

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

    def control_discrete_actions_only_right(self, action):
        t = self.car.car.get_transform()
        v = self.car.car.get_velocity()
        c = self.car.car.get_control()
        w = self.car.car.get_angular_velocity()
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
        target_speed = 30
        # throttle_low_limit = 0.0 #0.1 for low speeds
        brake = 0

        throttle = self.actions[action][0]
        steer = self.actions[action][1]
        # print(f"in STEP() {throttle = } and {steer = }")

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


#############################################################################################
#
# Trainer
#############################################################################################


class TrainerFollowLaneAutoCarlaSB3:
    """
    Mode: training
    Task: Follow Lane
    Algorithm: any of Stable-Baselines3
    Agent: Auto
    Simulator: Carla
    Framework: Stable-Baselines3
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

                ## --------------- RGB camera ---------------
                self.sensor_camera_rgb = CameraRGBSensor(self.new_car.car)
                self.actor_list.append(self.sensor_camera_rgb.sensor)

                ## --------------- RedMask Camera ---------------
                self.sensor_camera_red_mask = CameraRedMaskSemanticSensor(
                    self.new_car.car
                )
                self.actor_list.append(self.sensor_camera_red_mask.sensor)

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
                    self.sensor_camera_red_mask,
                    self.environment.environment,
                )
                # env = gym.make(self.env_params.env_name, **self.environment.environment)
                # check_env(env, warn=True)

                self.world.tick()

                ############################
                state, state_size = env.reset()
                # print(f"{state = } and {state_size = }")

                # self.world.tick()

                ######################################################################################
                #
                # STEPS
                ######################################################################################

                for step in tqdm(
                    range(1, 100),
                    ascii=True,
                    unit="episodes",
                ):
                    # print(f"\n{episode =} , {step = }")
                    # print(f"{state = } and {state_size = }")

                    self.world.tick()

                    if np.random.random() > epsilon:
                        # Get action from Q table
                        # action = np.argmax(agent_dqn.get_qs(state))
                        # print(f"\n{state =}")
                        # print(f"\n{np.array(state) =}")
                        # print(f"\n{type(np.array(state)) =}")
                        # print(f"\n{tf.convert_to_tensor(state) =}")
                        # print(f"\n{tf.convert_to_tensor(state)[0] =}")
                        # print(f"\n{tf.convert_to_tensor(state)[:] =}")

                        action = np.argmax(dqn_agent.get_qs(state))
                    else:
                        # Get random action
                        action = np.random.randint(
                            0, len(self.global_params.actions_set)
                        )

                    # print(f"{action = }\n")

                    ############################

                    new_state, reward, done, _ = env.step(action)
                    # print(f"{new_state = } and {reward = } and {done =}")

                    # Every step we update replay memory and train main network
                    # agent_dqn.update_replay_memory((state, action, reward, nextState, done))
                    dqn_agent.update_replay_memory(
                        (state, action, reward, new_state, done)
                    )
                    dqn_agent.train(done, step)

                    self.world.tick()

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
                        v=self.global_params.actions_set[action][
                            0
                        ],  # this case for discrete
                        w=self.global_params.actions_set[action][
                            1
                        ],  # this case for discrete
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

            # env.close()

        """
        #########################################################

        ## Reset env
        state, state_size = env.reset()

        print_messages(
            "main()",
            states=self.global_params.states,
            states_set=self.global_params.states_set,
            states_len=len(self.global_params.states_set),
            state_size=state_size,
            state=state,
            actions=self.global_params.actions,
            actions_set=self.global_params.actions_set,
            actions_len=len(self.global_params.actions_set),
            actions_range=range(len(self.global_params.actions_set)),
            batch_size=self.algoritmhs_params.batch_size,
            logs_tensorboard_dir=self.global_params.logs_tensorboard_dir,
            rewards=self.environment.environment["rewards"],
        )
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
         """

        """        
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

                env.world.tick()
                start_time_epoch = time.time()  # datetime.now()

                observation, _ = env.reset()
                while not done:
                    # print("after")
                    time_1 = time.time()
                    env.world.tick()
                    time_2 = time.time()
                    # print("before")
                    time_step = time_2 - time_1
                    print(f"{time_step =}")

                    if np.random.random() > epsilon:
                        # Get action from Q table
                        # action = np.argmax(agent_dqn.get_qs(state))
                        action = np.argmax(dqn_agent.get_qs(observation))
                    else:
                        # Get random action
                        action = np.random.randint(
                            0, len(self.global_params.actions_set)
                        )

                    start_step = time.time()
                    new_observation, reward, done, _ = env.step(action)
                    end_step = time.time()

                    # Every step we update replay memory and train main network
                    # agent_dqn.update_replay_memory((state, action, reward, nextState, done))
                    dqn_agent.update_replay_memory(
                        (observation, action, reward, new_observation, done)
                    )
                    dqn_agent.train(done, step)

                    self.global_params.time_steps[step] = end_step - start_step
                    cumulated_reward += reward
                    observation = new_observation
                    step += 1
        """
