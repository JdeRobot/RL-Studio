from collections import Counter
import copy
from datetime import datetime, timedelta
import math
import os
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


class FollowLaneStaticWeatherNoTrafficSB3(CarlaEnv):
    def __init__(self, **config):
        ## --------------- init env
        FollowLaneEnv.__init__(self, **config)
        ## --------------- init class variables
        FollowLaneCarlaConfig.__init__(self, **config)

        ## gym/stable-baselines3 interface

        ######## Actions Gym based
        print(f"{self.state_space = } and {self.action_space}")
        print(f"{len(self.actions) =}")
        print(f"{type(self.actions) =}")
        print(f"{self.actions =}")
        # Discrete Actions
        if self.action_space == "carla_discrete":
            self.action = spaces.Discrete(len(self.actions))
        else:  # Continuous Actions
            actions_to_array = np.array(
                [list(self.actions["v"]), list(self.actions["w"])]
            )
            print(f"{actions_to_array =}")
            print(f"{actions_to_array[0] =}")
            print(f"{actions_to_array[1] =}")
            self.action = spaces.Box(
                low=actions_to_array[0],
                high=actions_to_array[1],
                shape=(2,),
                dtype=np.float32,
            )
            print(f"{self.action.low = }")
            print(f"{self.action.high = }")
            print(f"{self.action.low[0] = }")
            print(f"{self.action.high[0] = }")

        print(f"action_space: {self.action}")

        ######## observations Gym based
        # image
        if self.state_space == "image":
            self.state = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )
        else:  # Discrete observations vector = [0.0, 3.6, 8.9]
            # TODO: change x_row for other list
            self.state = spaces.Discrete(len(self.x_row))  # temporary

        print(f"{self.state = }")
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
        self.client.set_timeout(3.0)

        # print(f"\n entre en DQN\n")

        self.world = self.client.load_world(config["town"])
        if config["town"] == "Town07_Opt":
            self.world.unload_map_layer(carla.MapLayer.Buildings)
            self.world.unload_map_layer(carla.MapLayer.Decals)
            self.world.unload_map_layer(carla.MapLayer.Foliage)
            self.world.unload_map_layer(carla.MapLayer.Particles)
            self.world.unload_map_layer(carla.MapLayer.Props)

        self.original_settings = self.world.get_settings()

        # TODO: si algo se jode hay que quitar esta linea
        # self.traffic_manager = self.client.get_trafficmanager(8000)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.1  # 0.05
        if config["sync"]:
            # TODO: si algo se jode hay que quitar esta linea
            # self.traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True  ####OJOOO
        else:
            # self.traffic_manager.set_synchronous_mode(False)
            settings.synchronous_mode = False  ###OJOJOOO

        self.world.apply_settings(settings)

        ### Weather
        weather = carla.WeatherParameters(
            cloudiness=70.0, precipitation=0.0, sun_altitude_angle=70.0
        )
        self.world.set_weather(weather)

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
        self.front_red_mask_camera = None

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
