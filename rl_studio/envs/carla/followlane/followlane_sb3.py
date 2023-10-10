from collections import Counter
import copy
from datetime import datetime, timedelta
import math
import os
import queue
import subprocess
import time
import sys

import carla
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
from gymnasium import spaces
import matplotlib.pyplot as plt
from memory_profiler import profile
import numpy as np
import pygame
import random
import weakref
import rospy
from sensor_msgs.msg import Image
import torch
import torchvision

from rl_studio.agents.utils import (
    print_messages,
    render_params,
)
from rl_studio.envs.carla.carla_env import CarlaEnv
from rl_studio.envs.carla.followlane.followlane_env import FollowLaneEnv
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils

from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
    CustomTimer,
)

from rl_studio.envs.carla.utils.global_route_planner import (
    GlobalRoutePlanner,
)

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


class FollowLaneStaticWeatherNoTrafficSB3(FollowLaneEnv):
    def __init__(self, **config):
        ## --------------- init env
        FollowLaneEnv.__init__(self, **config)
        ## --------------- init class variables
        FollowLaneCarlaConfig.__init__(self, **config)

        ################################## gym/stable-baselines3 interface

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

        ## --------------------------------------------------

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

        if self.sensor_camera_rgb is not None:
            self.destroy_all_actors_apply_batch()

        self.sensor_camera_rgb = None
        self.sensor_camera_red_mask = None

        self.actor_list = []

        # self.world.tick()

        ## ---  Car
        waypoints_town = self.world.get_map().generate_waypoints(5.0)
        init_waypoint = waypoints_town[self.waypoints_init + 1]
        filtered_waypoints = self.draw_waypoints(
            waypoints_town,
            self.waypoints_init,
            self.waypoints_target,
            self.waypoints_lane_id,
            2000,
        )
        self.target_waypoint = filtered_waypoints[-1]

        ## ---- random init position in the whole Town: actually for test functioning porposes
        if self.random_pose:
            self.setup_car_random_pose()
        ## -- Always same init position in a circuit predefined
        elif self.alternate_pose is False:
            self.setup_car_pose(self.start_alternate_pose, init=self.init_pose_number)

        ## -- Same circuit, but random init positions
        else:
            self.setup_car_pose(self.start_alternate_pose)

        ## -------------------------Sensors

        self.world.tick()

        ########### --- calculating STATES

        mask = self.preprocess_image(self.front_red_mask_camera)
        right_line_in_pixels = self.simplifiedperception.calculate_right_line(
            mask, self.x_row
        )

        pixels_in_state = mask.shape[1] / self.num_regions
        states = [
            int(value / pixels_in_state) for _, value in enumerate(right_line_in_pixels)
        ]
        # states = [5, 5, 5, 5]
        states_size = len(states)

        # print(f"{states =} and {states_size =}")
        return states, states_size

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


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get("fps", 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(
            carla.WorldSettings(
                # no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        print("yahora")
        print(f"{self.world = }")
        print(f"{self.world.tick() = }")
        self.frame = self.world.tick()
        print("paso por aqui")
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class XXXXXXFollowLaneStaticWeatherNoTrafficSB3(FollowLaneEnv):
    def __init__(self, **config):
        ## --------------- init env
        FollowLaneEnv.__init__(self, **config)
        ## --------------- init class variables
        FollowLaneCarlaConfig.__init__(self, **config)

        ################################## gym/stable-baselines3 interface

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

        ## --------------------------------------------------
        self.timer = CustomTimer()

        self.client = carla.Client(
            config["carla_server"],
            config["carla_client"],
        )
        # self.client = CarlaClient(
        #    config["carla_server"],
        #    config["carla_client"],
        # )

        # self.client.connect(connection_attempts=100)
        self.client.set_timeout(10.0)
        # print(f"\n maps in carla 0.9.13: {self.client.get_available_maps()}\n")

        # print(f"\n entre en DQN\n")

        # self.traffic_manager = self.client.get_trafficmanager(8000)

        if config["town"] == "Town07_Opt":
            self.world.unload_map_layer(carla.MapLayer.Buildings)
            self.world.unload_map_layer(carla.MapLayer.Decals)
            self.world.unload_map_layer(carla.MapLayer.Foliage)
            self.world.unload_map_layer(carla.MapLayer.Particles)
            self.world.unload_map_layer(carla.MapLayer.Props)

        # self.client.set_timeout(20.0)
        # self.original_settings = self.world.get_settings()

        self.world = self.client.load_world(config["town"])
        # self.world = self.client.get_world()
        # self.client.load_world(config["town"])

        # TODO: si algo se jode hay que quitar esta linea
        # self.traffic_manager = self.client.get_trafficmanager(8000)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 0.05
        self.world.apply_settings(settings)
        self.world.tick()

        # Set up the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)  # define TM seed for determinism

        # if config["sync"]:
        #    print(f"{config['sync'] =}")
        # TODO: si algo se jode hay que quitar esta linea
        #    settings.synchronous_mode = True  ####OJOOO
        # self.traffic_manager.set_synchronous_mode(True)
        # else:
        #    settings.synchronous_mode = False  ###OJOJOOO
        # self.traffic_manager.set_synchronous_mode(False)

        # self.world.apply_settings(settings)

        self.client.reload_world(False)

        # Set up the traffic manager
        # traffic_manager = self.client.get_trafficmanager(8000)
        # traffic_manager.set_synchronous_mode(True)
        # traffic_manager.set_random_device_seed(0)  # define TM seed for determinism

        ### Weather
        # weather = carla.WeatherParameters(
        #    cloudiness=70.0, precipitation=0.0, sun_altitude_angle=70.0
        # )
        # self.world.set_weather(weather)

        ## --------------- Blueprint ---------------
        self.blueprint_library = self.world.get_blueprint_library()
        ## --------------- Car ---------------
        self.vehicle = self.world.get_blueprint_library().filter("vehicle.*")[0]
        self.car = None
        ## --------------- collision sensor ---------------
        self.colsensor = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self.collision_hist = []
        self.col_sensor = None
        ## --------------- Lane invasion sensor ---------------
        self.laneinvsensor = self.world.get_blueprint_library().find(
            "sensor.other.lane_invasion"
        )
        self.lane_changing_hist = []
        self.lane_sensor = None
        ## --------------- Obstacle sensor ---------------
        self.obstsensor = self.world.get_blueprint_library().find(
            "sensor.other.obstacle"
        )
        self.obstacle_hist = []
        self.obstacle_sensor = None
        ## --------------- RGB camera ---------------
        self.rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{self.width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.height}")
        self.rgb_cam.set_attribute("fov", f"110")
        self.sensor_camera_rgb = None
        self.front_rgb_camera = None
        ## --------------- RedMask Camera ---------------
        self.red_mask_cam = self.world.get_blueprint_library().find(
            "sensor.camera.semantic_segmentation"
        )
        self.red_mask_cam.set_attribute("image_size_x", f"{self.width}")
        self.red_mask_cam.set_attribute("image_size_y", f"{self.height}")
        self.red_mask_cam.set_attribute("fov", f"110")
        self.sensor_camera_red_mask = None
        # self.front_red_mask_camera = None

        ## --------------- Segmentation Camera ---------------
        self.segm_cam = self.world.get_blueprint_library().find(
            "sensor.camera.semantic_segmentation"
        )
        self.segm_cam.set_attribute("image_size_x", f"{self.width}")
        self.segm_cam.set_attribute("image_size_y", f"{self.height}")
        self.sensor_camera_segmentation = None
        self.segmentation_cam = None

        ## --------------- more ---------------
        self.perfect_distance_pixels = None
        self.perfect_distance_normalized = None
        # self._control = carla.VehicleControl()
        self.params = {}
        self.target_waypoint = None

        self.spectator = self.world.get_spectator()
        # self.spectator = None
        self.actor_list = []
        self.is_finish = None
        self.dist_to_finish = None
        self.batch = []

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

        if self.sensor_camera_rgb is not None:
            self.destroy_all_actors_apply_batch()

        self.col_sensor = None
        self.sensor_camera_rgb = None
        self.sensor_camera_red_mask = None
        self.front_camera_bev = None
        self.front_camera_bev_mask = None
        self.sensor_camera_segmentation = None
        self.sensor_camera_lanedetector = None

        self.lane_sensor = None
        self.obstacle_sensor = None
        self.collision_hist = []
        self.lane_changing_hist = []
        self.obstacle_hist = []
        self.actor_list = []

        # self.world.tick()

        ## ---  Car
        waypoints_town = self.world.get_map().generate_waypoints(5.0)
        init_waypoint = waypoints_town[self.waypoints_init + 1]
        filtered_waypoints = self.draw_waypoints(
            waypoints_town,
            self.waypoints_init,
            self.waypoints_target,
            self.waypoints_lane_id,
            2000,
        )
        self.target_waypoint = filtered_waypoints[-1]

        ## ---- random init position in the whole Town: actually for test functioning porposes
        if self.random_pose:
            self.setup_car_random_pose()
        ## -- Always same init position in a circuit predefined
        elif self.alternate_pose is False:
            self.setup_car_pose(self.start_alternate_pose, init=self.init_pose_number)

        ## -- Same circuit, but random init positions
        else:
            self.setup_car_pose(self.start_alternate_pose)

        ## -------------------------Sensors
        ## --- Cameras

        ## ---- RGB
        # self.setup_rgb_camera()
        self.setup_rgb_camera_weakref()
        # self.setup_semantic_camera()
        while self.sensor_camera_rgb is None:  # or self.front_rgb_camera is None:
            # print(f"self.front_rgb_camera esperando sync {time.time()}")
            time.sleep(0.01)

        ## ---- Red Mask
        self.setup_red_mask_camera_weakref()
        while (
            self.sensor_camera_red_mask is None
        ):  # or self.front_red_mask_camera is None:
            # print(f"self.front_red_mask_camera esperando sync {time.time()}")
            time.sleep(0.01)

        ## --- SEgmentation camera
        self.setup_segmentation_camera_weakref()
        # self.setup_segmentation_camera()
        while self.sensor_camera_segmentation is None:
            time.sleep(0.01)

        # ## ---- LAneDetector
        # self.setup_lane_detector_camera_weakref()
        # while self.sensor_camera_lane_detector is None:
        #    time.sleep(0.01)

        ## --- Detectors Sensors
        self.setup_lane_invasion_sensor_weakref()

        # time.sleep(1)
        self.world.tick()

        # self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        ########### --- calculating STATES
        # print("time.sleep(4)")
        # time.sleep(4)
        # print("salimos time.sleep(4)")

        mask = self.preprocess_image(self.front_red_mask_camera)
        right_line_in_pixels = self.simplifiedperception.calculate_right_line(
            mask, self.x_row
        )

        pixels_in_state = mask.shape[1] / self.num_regions
        states = [
            int(value / pixels_in_state) for _, value in enumerate(right_line_in_pixels)
        ]
        # states = [5, 5, 5, 5]
        states_size = len(states)

        # print(f"{states =} and {states_size =}")
        return states, states_size

        """
        print("estoy aqui en reset()")
        with CarlaSyncMode(self.world, self.sensor_camera_rgb, fps=30) as sync_mode:
            # Advance the simulation and wait for the data.
            print(f"entro")
            snapshot, image_rgb = sync_mode.tick(timeout=2.0)
            print("Aquieee")
            fps = round(1.0 / snapshot.timestamp.delta_seconds)
            print(f"{fps = }")

            time.sleep(1)

            self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

            ########### --- calculating STATES
            print("time.sleep(4)")
            time.sleep(4)
            print("salimos time.sleep(4)")

            mask = self.simplifiedperception.preprocess_image(
                self.front_red_mask_camera
            )
            right_line_in_pixels = self.simplifiedperception.calculate_right_line(
                mask, self.x_row
            )

            pixels_in_state = mask.shape[1] / self.num_regions
            states = [
                int(value / pixels_in_state)
                for _, value in enumerate(right_line_in_pixels)
            ]
            # states = [5, 5, 5, 5]
            states_size = len(states)

            # print(f"{states =} and {states_size =}")
        return states, states_size
        """

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

    ###########################################
    #
    # Step
    #
    ###########################################

    def step(self, action):
        """
        state: vector / simplified percepction
        actions: discrete
        only right line
        """

        self.control_discrete_actions_only_right(action)

        ########### --- calculating STATES
        time.sleep(0.1)
        mask = self.simplifiedperception.preprocess_image(self.front_red_mask_camera)

        ########### --- Calculating center ONLY with right line
        right_line_in_pixels = self.simplifiedperception.calculate_right_line(
            mask, self.x_row
        )

        image_center = mask.shape[1] // 2
        dist = [
            image_center - right_line_in_pixels[i]
            for i, _ in enumerate(right_line_in_pixels)
        ]
        dist_normalized = [
            float((image_center - right_line_in_pixels[i]) / image_center)
            for i, _ in enumerate(right_line_in_pixels)
        ]

        pixels_in_state = mask.shape[1] // self.num_regions
        state_right_lines = [
            i for i in range(1, mask.shape[1]) if i % pixels_in_state == 0
        ]

        ## STATES

        states = [
            int(value / pixels_in_state) for _, value in enumerate(right_line_in_pixels)
        ]

        ## -------- Ending Step()...
        done = False
        ground_truth_normal_values = [
            correct_normalized_distance[value] for i, value in enumerate(self.x_row)
        ]

        reward, done = self.autocarlarewards.rewards_sigmoid_only_right_line(
            dist_normalized, ground_truth_normal_values
        )

        ## -------- ... or Finish by...
        ground_truth_pixel_values = [
            correct_pixel_distance[value] for i, value in enumerate(self.x_row)
        ]

        AutoCarlaUtils.show_image_only_right_line(
            "front RGB",
            self.front_rgb_camera[(self.front_rgb_camera.shape[0] // 2) :],
            1,
            right_line_in_pixels,
            ground_truth_pixel_values,
            dist_normalized,
            [4, 4, 4, 4],
            self.x_row,
            1250,
            10,
            state_right_lines,
        )

        return states, reward, done, {}

    ##################################################
    #
    #   Control
    ###################################################

    def control_discrete_actions_only_right(self, action):
        t = self.car.get_transform()
        v = self.car.get_velocity()
        c = self.car.get_control()
        w = self.car.get_angular_velocity()
        a = self.car.get_acceleration()
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

        if curr_speed > target_speed:
            throttle = 0.45

        self.car.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

    ########################################################
    #  utils
    #
    ########################################################

    def __del__(self):
        print("__del__ called")

    ##################
    #
    # Destroyers
    ##################

    def destroy_all_actors_apply_batch(self):
        print("\ndestroying %d actors with apply_batch method" % len(self.actor_list))
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.actor_list[::-1]]
        )

    def destroy_all_actors(self):
        print("\ndestroying %d actors with destroy() method" % len(self.actor_list))
        for actor in self.actor_list[::-1]:
            actor.destroy()

    ########################################################
    #  waypoints, car pose
    #
    ########################################################

    def draw_waypoints(self, spawn_points, init, target, lane_id, life_time):
        """
        WArning: target always has to be the last waypoint, since it is the FINISH
        """
        filtered_waypoints = []
        i = init
        for waypoint in spawn_points[init + 1 : target + 2]:
            filtered_waypoints.append(waypoint)
            string = f"[{waypoint.road_id},{waypoint.lane_id},{i}]"
            if waypoint.lane_id == lane_id:
                if i != target:
                    self.world.debug.draw_string(
                        waypoint.transform.location,
                        f"X - {string}",
                        draw_shadow=False,
                        color=carla.Color(r=0, g=255, b=0),
                        life_time=life_time,
                        persistent_lines=True,
                    )
                else:
                    self.world.debug.draw_string(
                        waypoint.transform.location,
                        f"X - {string}",
                        draw_shadow=False,
                        color=carla.Color(r=255, g=0, b=0),
                        life_time=life_time,
                        persistent_lines=True,
                    )
            i += 1

        return filtered_waypoints

    # @profile
    def setup_car_pose(self, init_positions, init=None):
        if init is None:
            pose_init = np.random.randint(0, high=len(init_positions))
        else:
            pose_init = init

        # print(f"{pose_init =}")
        # print(f"{random_init = }")
        # print(f"{self.start_alternate_pose = }")
        # print(f"{self.start_alternate_pose[random_init][0] = }")
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

        """ these is for apply_batch_sync"""
        # self.car = carla.command.SpawnActor(self.vehicle, location)
        # while self.car is None:
        #    self.car = carla.command.SpawnActor(self.vehicle, location)

        # self.batch.append(self.car)

        """ original """
        self.car = self.world.spawn_actor(self.vehicle, location)
        while self.car is None:
            self.car = self.world.spawn_actor(self.vehicle, location)

        # self.batch.append(self.car)

        self.actor_list.append(self.car)

        time.sleep(1)

    def setup_car_alternate_pose(self, init_positions):
        random_init = np.random.randint(0, high=len(init_positions))
        # print(f"{random_init = }")
        # print(f"{self.start_alternate_pose = }")
        # print(f"{self.start_alternate_pose[random_init][0] = }")
        location = carla.Transform(
            carla.Location(
                x=self.start_alternate_pose[random_init][0],
                y=self.start_alternate_pose[random_init][1],
                z=self.start_alternate_pose[random_init][2],
            ),
            carla.Rotation(
                pitch=self.start_alternate_pose[random_init][3],
                yaw=self.start_alternate_pose[random_init][4],
                roll=self.start_alternate_pose[random_init][5],
            ),
        )

        self.car = self.world.spawn_actor(self.vehicle, location)
        while self.car is None:
            self.car = self.world.spawn_actor(self.vehicle, location)

        self.actor_list.append(self.car)
        time.sleep(1)

    def setup_car_fix_pose(self, init):
        """
        Town07, road 20, -1, init curves
        (73.7, -10, 0.3, 0.0, -62.5, 0.0)
        """

        """
        # car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = carla.Transform(
            carla.Location(
                x=init.transform.location.x,
                y=init.transform.location.y,
                z=init.transform.location.z,
            ),
            carla.Rotation(
                pitch=init.transform.rotation.pitch,
                yaw=init.transform.rotation.yaw,
                roll=init.transform.rotation.roll,
            ),
        )
        
        print(f"{init.transform.location.x = }")
        print(f"{init.transform.location.y = }")
        print(f"{init.lane_id = }")
        print(f"{init.road_id = }")
        print(f"{init.s = }")
        print(f"{init.id = }")
        print(f"{init.lane_width = }")
        print(f"{init.lane_change = }")
        print(f"{init.lane_type = }")
        print(f"{init.right_lane_marking = }")
        """

        location = carla.Transform(
            carla.Location(x=73.7, y=-10, z=0.300000),
            carla.Rotation(pitch=0.000000, yaw=-62.5, roll=0.000000),
        )

        self.car = self.world.spawn_actor(self.vehicle, location)
        while self.car is None:
            self.car = self.world.spawn_actor(self.vehicle, location)

        self.actor_list.append(self.car)
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

    ##################################################################
    #
    #  Sensors
    ##################################################################

    #######################################################
    #
    # Detectors
    #######################################################

    # @profile
    def setup_col_sensor_weakref(self):
        """weakref"""
        # colsensor = self.world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.col_sensor = self.world.spawn_actor(
            self.colsensor, transform, attach_to=self.car
        )
        self.actor_list.append(self.col_sensor)
        weak_self = weakref.ref(self)
        self.col_sensor.listen(
            lambda event: FollowLaneStaticWeatherNoTrafficSB3._collision_data(
                weak_self, event
            )
        )
        # self.col_sensor.listen(self.collision_data)

    @staticmethod
    def _collision_data(weak_self, event):
        """weakref"""
        self = weak_self()
        if not self:
            return

        self.collision_hist.append(event)
        actor_we_collide_against = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        print(
            f"you have crashed with {actor_we_collide_against.type_id} with impulse {impulse} and intensity {intensity}"
        )

    def setup_col_sensor(self):
        # colsensor = self.world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.col_sensor = self.world.spawn_actor(
            self.colsensor, transform, attach_to=self.car
        )
        self.actor_list.append(self.col_sensor)
        # self.col_sensor.listen(lambda event: self.collision_data(event))
        self.col_sensor.listen(self.collision_data)

    def collision_data(self, event):
        self.collision_hist.append(event)
        actor_we_collide_against = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        print(
            f"you have crashed with {actor_we_collide_against.type_id} with impulse {impulse} and intensity {intensity}"
        )
        # self.actor_list.append(actor_we_collide_against)

    def setup_lane_invasion_sensor_weakref(self):
        """weakref"""
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.lane_sensor = self.world.spawn_actor(
            self.laneinvsensor, transform, attach_to=self.car
        )

        # self.lane_sensor = carla.command.SpawnActor(
        #    self.laneinvsensor, transform, parent=self.car
        # )

        # self.batch.append(self.lane_sensor)
        self.actor_list.append(self.lane_sensor)
        # self.col_sensor.listen(lambda event: self.collision_data(event))
        weak_self = weakref.ref(self)
        # self.lane_sensor.listen(self.lane_changing_data)
        self.lane_sensor.listen(
            lambda event: FollowLaneStaticWeatherNoTrafficSB3._lane_changing(
                weak_self, event
            )
        )

    @staticmethod
    def _lane_changing(weak_self, event):
        """weakref"""
        self = weak_self()
        if not self:
            return
        self.lane_changing_hist.append(event)
        # print(f"you have changed the lane")

    def setup_lane_invasion_sensor(self):
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.lane_sensor = self.world.spawn_actor(
            self.laneinvsensor, transform, attach_to=self.car
        )
        self.actor_list.append(self.lane_sensor)
        # self.col_sensor.listen(lambda event: self.collision_data(event))
        self.lane_sensor.listen(self.lane_changing_data)

    def lane_changing_data(self, event):
        self.lane_changing_hist.append(event)
        print(f"you have changed the lane")

    def setup_obstacle_sensor_weakref(self):
        """weakref"""
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.obstacle_sensor = self.world.spawn_actor(
            self.obstsensor, transform, attach_to=self.car
        )
        self.actor_list.append(self.obstacle_sensor)
        # self.col_sensor.listen(lambda event: self.collision_data(event))
        weak_self = weakref.ref(self)
        # self.obstacle_sensor.listen(self.obstacle_data)
        self.obstacle_sensor.listen(
            lambda event: FollowLaneStaticWeatherNoTrafficSB3._obstacle_sensor(
                weak_self, event
            )
        )

    @staticmethod
    def _obstacle_sensor(weak_self, event):
        """weakref"""
        self = weak_self()
        if not self:
            return
        self.obstacle_hist.append(event)
        print(f"you have found an obstacle")

    def setup_obstacle_sensor(self):
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.obstacle_sensor = self.world.spawn_actor(
            self.obstsensor, transform, attach_to=self.car
        )
        self.actor_list.append(self.obstacle_sensor)
        # self.col_sensor.listen(lambda event: self.collision_data(event))
        self.obstacle_sensor.listen(self.obstacle_data)

    def obstacle_data(self, event):
        self.obstacle_hist.append(event)
        print(f"you have found an obstacle")

    #######################################
    #
    # spectator
    #######################################
    def setup_spectator(self):
        # self.spectator = self.world.get_spectator()
        car_transfor = self.car.get_transform()
        # world_snapshot = self.world.wait_for_tick()
        # actor_snapshot = world_snapshot.find(self.car.id)
        # Set spectator at given transform (vehicle transform)
        # self.spectator.set_transform(actor_snapshot.get_transform())
        self.spectator.set_transform(
            carla.Transform(
                car_transfor.location + carla.Location(z=60),
                carla.Rotation(pitch=-90, roll=-90),
            )
        )
        # self.actor_list.append(self.spectator)

    ######################################################
    #
    # Cameras
    ######################################################

    ########################################
    #
    # RGB
    ########################################

    # @profile
    def setup_rgb_camera_weakref(self):
        """weakref"""

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.sensor_camera_rgb = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.car
        )

        # self.sensor_camera_rgb = carla.command.SpawnActor(
        #    self.rgb_cam, transform, parent=self.car
        # )

        # self.batch.append(self.sensor_camera_rgb)

        self.actor_list.append(self.sensor_camera_rgb)
        # print(f"{len(self.actor_list) = }")
        weak_self = weakref.ref(self)
        # self.sensor_camera_rgb.listen(self.save_rgb_image)
        self.sensor_camera_rgb.listen(
            lambda event: FollowLaneStaticWeatherNoTrafficSB3._rgb_image(
                weak_self, event
            )
        )

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

    def setup_rgb_camera(self):
        # print("enter setup_rg_camera")
        # rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
        # rgb_cam.set_attribute("image_size_x", f"{self.width}")
        # rgb_cam.set_attribute("image_size_y", f"{self.height}")
        # rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.sensor_camera_rgb = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.sensor_camera_rgb)
        # print(f"{len(self.actor_list) = }")
        self.sensor_camera_rgb.listen(self.save_rgb_image)

    def save_rgb_image(self, data: carla.Image):
        if not isinstance(data, carla.Image):
            raise ValueError("Argument must be a carla.Image")
        image = np.array(data.raw_data)
        image = image.reshape((480, 640, 4))
        image = image[:, :, :3]
        # self._data_dict["image"] = image3
        self.front_rgb_camera = image

    def _save_rgb_image(self, image):
        # t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # if self.display_man.render_enabled():
        #    self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # t_end = self.timer.time()
        # self.time_processing += t_end - t_start
        # self.tics_processing += 1
        # cv2.imshow("", array)
        # cv2.waitKey(1)
        self.front_rgb_camera = array

    ##################
    #
    # Red Mask Camera
    ##################
    def setup_red_mask_camera_weakref(self):
        """weakref"""

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.sensor_camera_red_mask = self.world.spawn_actor(
            self.red_mask_cam, transform, attach_to=self.car
        )

        # self.sensor_camera_red_mask = carla.command.SpawnActor(
        #    self.red_mask_cam, transform, parent=self.car
        # )

        # self.batch.append(self.sensor_camera_red_mask)
        self.actor_list.append(self.sensor_camera_red_mask)
        weak_self = weakref.ref(self)

        # start_camera_red_mask = time.time()
        self.sensor_camera_red_mask.listen(
            lambda event: FollowLaneStaticWeatherNoTrafficSB3._red_mask_semantic_image(
                weak_self, event
            )
        )
        # end_camera_red_mask = time.time()
        # print(
        #    f"\n====> sensor_camera_red_mask time processing: {end_camera_red_mask - start_camera_red_mask = }"
        # )

        # self.sensor_camera_red_mask.listen(self.save_red_mask_semantic_image)

    @staticmethod
    def _red_mask_semantic_image(weak_self, image):
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

    def setup_red_mask_camera(self):
        # self.red_mask_cam = self.world.get_blueprint_library().find(
        #    "sensor.camera.semantic_segmentation"
        # )
        # self.red_mask_cam.set_attribute("image_size_x", f"{self.width}")
        # self.red_mask_cam.set_attribute("image_size_y", f"{self.height}")

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.sensor_camera_red_mask = self.world.spawn_actor(
            self.red_mask_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.sensor_camera_red_mask)
        self.sensor_camera_red_mask.listen(self.save_red_mask_semantic_image)

    def save_red_mask_semantic_image(self, image):
        # t_start = self.timer.time()
        # print(f"en red_mask calbback")
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        hsv_nemo = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

        if (
            self.world.get_map().name == "Carla/Maps/Town07"
            or self.world.get_map().name == "Carla/Maps/Town04"
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

    ##################
    #
    # BEV
    ##################
    def setup_bev_camera_weakref(self):
        """weakref"""

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.sensor_bev_camera = self.world.spawn_actor(
            self.bev_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.sensor_bev_camera)
        # print(f"{len(self.actor_list) = }")
        weak_self = weakref.ref(self)
        # self.sensor_bev_camera.listen(self.save_bev_image)
        self.sensor_bev_camera.listen(
            lambda event: FollowLaneStaticWeatherNoTrafficSB3._bev_image(
                weak_self, event
            )
        )

    @staticmethod
    def _bev_image(weak_self, image):
        """weakref"""
        self = weak_self()
        if not self:
            return

        car_bp = self.world.get_actors().filter("vehicle.*")[0]
        # car_bp = self.vehicle
        birdview = self.birdview_producer.produce(
            agent_vehicle=car_bp  # carla.Actor (spawned vehicle)
        )
        image = BirdViewProducer.as_rgb(birdview)
        hsv_nemo = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Extract central line
        light_center_line = (17, 134, 232)
        dark_center_line = (19, 136, 234)
        mask_center_line = cv2.inRange(hsv_nemo, light_center_line, dark_center_line)
        result = cv2.bitwise_and(hsv_nemo, hsv_nemo, mask=mask_center_line)

        # image = np.rot90(image)
        image = np.array(image)
        result = np.array(result)
        if image.shape[0] != image.shape[1]:
            if image.shape[0] > image.shape[1]:
                difference = image.shape[0] - image.shape[1]
                extra_left, extra_right = int(difference / 2), int(difference / 2)
                extra_top, extra_bottom = 0, 0
            else:
                difference = image.shape[1] - image.shape[0]
                extra_left, extra_right = 0, 0
                extra_top, extra_bottom = int(difference / 2), int(difference / 2)
            image = np.pad(
                image,
                ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            image = np.pad(
                image,
                ((100, 100), (50, 50), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        self.front_camera_bev = image
        self.front_camera_bev_mask = result

    def setup_bev_camera(self):
        # print("enter setup_rg_camera")
        # rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
        # rgb_cam.set_attribute("image_size_x", f"{self.width}")
        # rgb_cam.set_attribute("image_size_y", f"{self.height}")
        # rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.sensor_bev_camera = self.world.spawn_actor(
            self.bev_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.sensor_bev_camera)
        # print(f"{len(self.actor_list) = }")
        self.sensor_bev_camera.listen(self.save_bev_image)

    def save_bev_image(self, image):
        car_bp = self.world.get_actors().filter("vehicle.*")[0]
        # car_bp = self.vehicle
        birdview = self.birdview_producer.produce(
            agent_vehicle=car_bp  # carla.Actor (spawned vehicle)
        )
        image = BirdViewProducer.as_rgb(birdview)
        hsv_nemo = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Extract central line
        light_center_line = (17, 134, 232)
        dark_center_line = (19, 136, 234)
        mask_center_line = cv2.inRange(hsv_nemo, light_center_line, dark_center_line)
        result = cv2.bitwise_and(hsv_nemo, hsv_nemo, mask=mask_center_line)

        # image = np.rot90(image)
        image = np.array(image)
        result = np.array(result)
        if image.shape[0] != image.shape[1]:
            if image.shape[0] > image.shape[1]:
                difference = image.shape[0] - image.shape[1]
                extra_left, extra_right = int(difference / 2), int(difference / 2)
                extra_top, extra_bottom = 0, 0
            else:
                difference = image.shape[1] - image.shape[0]
                extra_left, extra_right = 0, 0
                extra_top, extra_bottom = int(difference / 2), int(difference / 2)
            image = np.pad(
                image,
                ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            image = np.pad(
                image,
                ((100, 100), (50, 50), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        self.front_camera_bev = image
        self.front_camera_bev_mask = result

    ##################
    #
    # Segmentation Camera
    ##################

    def setup_segmentation_camera_weakref(self):
        """weakref"""

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.sensor_camera_segmentation = self.world.spawn_actor(
            self.segm_cam, transform, attach_to=self.car
        )

        # self.sensor_camera_segmentation = carla.command.SpawnActor(
        #    self.segm_cam, transform, parent=self.car
        # )

        # self.batch.append(self.sensor_camera_segmentation)
        self.actor_list.append(self.sensor_camera_segmentation)
        # print(f"{len(self.actor_list) = }")
        weak_self = weakref.ref(self)
        # self.sensor_camera_segmentation.listen(self.save_segmentation_image)
        self.sensor_camera_segmentation.listen(
            lambda event: FollowLaneStaticWeatherNoTrafficSB3._segmentation_image(
                weak_self, event
            )
        )

    @staticmethod
    def _segmentation_image(weak_self, image):
        """weakref"""
        self = weak_self()
        if not self:
            return

        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.segmentation_cam = array

    def setup_segmentation_camera(self):
        # print("enter setup_rg_camera")
        # rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
        # rgb_cam.set_attribute("image_size_x", f"{self.width}")
        # rgb_cam.set_attribute("image_size_y", f"{self.height}")
        # rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.sensor_camera_segmentation = self.world.spawn_actor(
            self.segm_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.sensor_camera_segmentation)
        # print(f"{len(self.actor_list) = }")
        self.sensor_camera_segmentation.listen(self.save_segmentation_image)

    def save_segmentation_image(self, image: carla.Image):
        # t_start = self.timer.time()

        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        self.segmentation_cam = array
        # if self.display_man.render_enabled():
        #    self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # t_end = self.timer.time()
        # self.time_processing += t_end - t_start
        # self.tics_processing += 1
