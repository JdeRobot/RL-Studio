from collections import Counter, OrderedDict
from datetime import datetime, timedelta
import glob
from statistics import median
import os
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
from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils
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
from stable_baselines3.common.env_checker import check_env

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


class CarlaEnv(gym.Env):
    def __init__(self, **config):
        super(CarlaEnv, self).__init__()
        ## --------------- init env
        # FollowLaneEnv.__init__(self, **config)
        ## --------------- init class variables
        # FollowLaneCarlaConfig.__init__(self, **config)

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

        print(f"{self.alternate_pose =}\n")
        print(f"{self.start_alternate_pose =}\n")

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
        # print(f"{self.state.high = }")
        # print(f"{self.state.low[0] = }")
        # print(f"{self.state.high[0] = }")

        # Vector State

        #########################################################################
        # ## --------------------------------------------------

        """
        self.client = carla.Client(
            config["carla_server"],
            config["carla_client"],
        )
        self.client.set_timeout(10.0)
        print(f"\n maps in carla 0.9.13: {self.client.get_available_maps()}\n")

        self.world = self.client.load_world(config["town"])

        # Sync mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 0.05
        self.world.apply_settings(settings)

        # Set up the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)  # define TM seed for determinism

        self.client.reload_world(False)

        print(f"{settings.synchronous_mode =}")
        self.world.tick()
        # print(f"{self.world.tick()=}")

        # Town07 take layers off
        if config["town"] == "Town07_Opt":
            self.world.unload_map_layer(carla.MapLayer.Buildings)
            self.world.unload_map_layer(carla.MapLayer.Decals)
            self.world.unload_map_layer(carla.MapLayer.Foliage)
            self.world.unload_map_layer(carla.MapLayer.Particles)
            self.world.unload_map_layer(carla.MapLayer.Props)

        """

        self.params = config

        self.world = self.params["world"]
        # self.actor_list = self.params["actor_list"]
        self.client = self.params["client"]
        self.vehicle = self.params["vehicle"]
        self.car = self.params["car"]
        self.front_red_mask_camera = self.params["front_red_mask_camera"]
        print(f"{self.front_red_mask_camera=}")
        # self.car = None
        # self.actor_list = []
        # self.front_rgb_camera = None
        # self.front_red_mask_camera = None

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

        # self.client.apply_batch(
        #    [carla.command.DestroyActor(x) for x in self.actor_list[::-1]]
        # )

        ############################################################################
        ## --------------- Car ---------------
        ## ---- random init position in the whole Town: actually for test functioning purposes
        if self.random_pose:
            self.setup_car_random_pose()
        ## -- Always same init position in a circuit predefined
        elif self.alternate_pose is False:
            self.setup_car_pose(self.start_alternate_pose, init=self.init_pose_number)

        ## -- Same circuit, but random init positions
        else:
            # self.setup_car_alternate_pose(self.start_alternate_pose)
            self.setup_car_pose(self.start_alternate_pose)

        ############################################################################

        ############################################################################
        ########### --- calculating STATES
        # while self.front_red_mask_camera is None:
        #    print(f"{self.front_red_mask_camera = }")
        #    time.sleep(1)

        # mask = self.preprocess_image(self.front_red_mask_camera)
        mask = self.preprocess_image(self.front_red_mask_camera)
        right_line_in_pixels = self.calculate_right_line(mask, self.x_row)

        pixels_in_state = mask.shape[1] / self.num_regions
        states = [
            int(value / pixels_in_state) for _, value in enumerate(right_line_in_pixels)
        ]
        # states = [5, 5, 5, 5]
        states_size = len(states)

        # print(f"{states =} and {states_size =}")
        return states, states_size

    #####################################################################################
    #
    #   methods
    #####################################################################################

    def setup_car_pose(self, init_positions, init=None):
        if init is None:
            pose_init = np.random.randint(0, high=len(init_positions))
        else:
            pose_init = init

        location = carla.Transform(
            carla.Location(
                x=self.start_alternate_pose[pose_init][0],
                y=self.start_alternate_pose[pose_init][1],
                z=self.start_alternate_pose[pose_init][2],
            ),
            carla.Rotation(
                pitch=self.start_alternate_pose[pose_init][3],
                yaw=self.start_alternate_pose[pose_init][4],
                roll=self.start_alternate_pose[pose_init][5],
            ),
        )

        """ original """
        self.car = self.world.spawn_actor(self.vehicle, location)
        while self.car is None:
            self.car = self.world.spawn_actor(self.vehicle, location)

        # self.actor_list.append(self.car)

        time.sleep(1)

    def setup_car_random_pose(self):
        # car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(self.vehicle, location)
        while self.car is None:
            self.car = self.world.try_spawn_actor(self.vehicle, location)

        # self.batch.append(self.car)
        self.actor_list.append(self.car)
        time.sleep(1)

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
        # print(f"{inv_index_right = }")
        # offset = 10
        # inv_index_right_plus_offset = [
        #    inv_index_right[x] + offset if inv_index_right[x] != 0 else 0
        #    for x, _ in enumerate(inv_index_right)
        # ]
        # print(f"{inv_index_right = }")
        # index_right = [
        #    mask.shape[1] - inv_index_right_plus_offset[x]
        #    if inv_index_right_plus_offset[x] != 0
        #    else 0
        #    for x, _ in enumerate(inv_index_right_plus_offset)
        # ]
        index_right = [
            mask.shape[1] - inv_index_right[x] if inv_index_right[x] != 0 else 0
            for x, _ in enumerate(inv_index_right)
        ]

        return index_right


# ==============================================================================
# -- RGB CameraManager -------------------------------------------------------------
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
        if self.front_red_mask_camera is not None:
            #    time.sleep(0.01)
            #    print(f"self.front_red_mask_camera leyendo")
            print(f"in callback() {time.time()}")


# ==============================================================================
# -- New Car -------------------------------------------------------------
# ==============================================================================
class NewCar(object):
    def __init__(self, parent_actor, alternate_pose, start_alternate_pose):
        self.car = None
        self._parent = parent_actor

        # self.world = self._parent.get_world()
        self.world = self._parent
        vehicle = self.world.get_blueprint_library().filter("vehicle.*")[0]
        if alternate_pose is None:
            pose_init = np.random.randint(0, high=len(start_alternate_pose))
        else:
            pose_init = alternate_pose

        location = carla.Transform(
            carla.Location(
                x=start_alternate_pose[pose_init][0],
                y=start_alternate_pose[pose_init][1],
                z=start_alternate_pose[pose_init][2],
            ),
            carla.Rotation(
                pitch=start_alternate_pose[pose_init][3],
                yaw=start_alternate_pose[pose_init][4],
                roll=start_alternate_pose[pose_init][5],
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
    #
    # Main
    #########################################################################

    def main(self):
        """ """

        #########################################################################
        #
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

        try:
            #########################################################################
            #
            # Vars
            #########################################################################
            print(f"\n al inicio de Main() {self.environment.environment =}\n")
            self.client = carla.Client(
                self.environment.environment["carla_server"],
                self.environment.environment["carla_client"],
            )
            self.client.set_timeout(5.0)
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

            print(f"{settings.synchronous_mode =}")
            self.world.tick()
            # print(f"{self.world.tick()=}")

            # Town07 take layers off
            if self.environment.environment["town"] == "Town07_Opt":
                self.world.unload_map_layer(carla.MapLayer.Buildings)
                self.world.unload_map_layer(carla.MapLayer.Decals)
                self.world.unload_map_layer(carla.MapLayer.Foliage)
                self.world.unload_map_layer(carla.MapLayer.Particles)
                self.world.unload_map_layer(carla.MapLayer.Props)

            ######################### car, sensors
            #
            #
            ###########################################
            # actor_list = []
            ## --------------- Blueprint ---------------

            self.new_car = NewCar(
                self.world,
                self.environment.environment["alternate_pose"],
                self.environment.environment["start_alternate_pose"],
            )
            self.actor_list.append(self.new_car.car)

            ############################################
            #
            ## --------------- Sensors ---------------
            #################################

            ## --------------- RGB camera ---------------
            self.sensor_camera_rgb = CameraRGBSensor(self.new_car.car)
            self.actor_list.append(self.sensor_camera_rgb.sensor)

            ## --------------- RedMask Camera ---------------
            self.sensor_camera_red_mask = CameraRedMaskSemanticSensor(self.new_car.car)
            self.actor_list.append(self.sensor_camera_red_mask.sensor)

            self.world.tick()

            ############################################################################

            print(
                f"in main() {self.sensor_camera_rgb.front_rgb_camera} in {time.time()}"
            )
            print(
                f"in main() {self.sensor_camera_red_mask.front_red_mask_camera} in {time.time()}"
            )

            # env = CarlaEnv(**self.environment.environment)
            # env = gym.make(self.env_params.env_name, **self.environment.environment)
            # check_env(env, warn=True)

            ############################################################################
            #
            # RESET, AGENT, FOR
            ############################################################################

            ## Reset env
            # self.world.tick()
            # state, state_size = env.reset()
            # print(
            #    f"Again in main() {self.sensor_camera_red_mask.front_red_mask_camera} in {time.time()}"
            # )

            """
            dentro del for, previo al reset posicionaos el car
            """

            while True:
                self.world.tick()
                if self.sensor_camera_rgb.front_rgb_camera is not None:
                    print(
                        f"Again in main() self.sensor_camera_rgb.front_rgb_camera in {time.time()}"
                    )
                if self.sensor_camera_red_mask.front_red_mask_camera is not None:
                    print(f"Again in main() in {time.time()}")

                AutoCarlaUtils.show_image(
                    "RGB",
                    self.sensor_camera_rgb.front_rgb_camera,
                    1200,
                    400,
                )
                AutoCarlaUtils.show_image(
                    "segmentation_cam",
                    self.sensor_camera_red_mask.front_red_mask_camera,
                    400,
                    400,
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
                print(f"enro?")

            # destroy_all_actors()
            for actor in self.actor_list[::-1]:
                actor.destroy()

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
