import glob
import os
import random
import sys
import time
import weakref

import carla
from cv_bridge import CvBridge
import cv2
import numpy as np
import pygame
from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rl_studio.envs.carla.carla_env import CarlaEnv

# from rl_studio.envs.carla.utils.bounding_boxes import ClientSideBoundingBoxes
# from rl_studio.envs.carla.utils.logger import logger
# from rl_studio.envs.carla.utils.weather import Weather
from rl_studio.envs.carla.utils.environment import (
    apply_sun_presets,
    apply_weather_presets,
    apply_weather_values,
    apply_lights_to_cars,
    apply_lights_manager,
    get_args,
)


VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90
BB_COLOR = (248, 64, 24)

IM_WIDTH = 640
IM_HEIGHT = 480


class FollowLaneEnv(CarlaEnv):
    def __init__(self, **config):
        """Constructor of the class."""

        print(f"in FollowLaneEnv -> launching CarlaEnv\n")
        # CarlaEnv.__init__(self, **config)
        print(f"\nin FollowLaneEnv again\n")
        print(f"{config=}\n")
        # self.actor_list = []
        # self.carla_map = None
        # self.client = None
        # self.world = None
        # self.transform = None
        # self.vehicle = None
        # self.camera = None
        # self.vehicle = self.blueprint_library.filter(config["car"])[0]
        # self.height_image = config["height_image"]
        # self.width_image = config["width_image"]
        # self.vehicles = None

        # self.display = None
        # self.image = None
        # self.capture = True
        # pygame.init()
        # pygame.font.init()
        # world = None
        # original_settings = None

        # ----------------------------
        # launch client and world
        # ----------------------------
        # self.client = carla.Client('localhost', 2000)
        # self.client = carla.Client(config["carla_server"], config["carla_client"])
        # self.client.set_timeout(2.0)
        # self.world = self.client.get_world()
        # self.blueprint_library = self.world.get_blueprint_library()
        # self.car_model = random.choice(self.blueprint_library.filter("vehicle.*.*"))
        # self.car_model = self.blueprint_library.filter('model3')[0]

        # ----------------------------
        # Weather: Static
        # Traffic and pedestrians: No
        # ----------------------------
        # self.weather = config["weather"]
        # self.traffic = config["traffic_pedestrians"]
        # if self.weather != "dynamic" and self.traffic is False:
        #    pass

        # self.actor_list = []

    #########################################################################

    def reset_(self):
        self.actor_list = []
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()

        bp = blueprint_library.filter("model3")[0]
        print(bp)

        spawn_point = random.choice(world.get_map().get_spawn_points())

        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        self.actor_list.append(vehicle)

        # sleep for 5 seconds, then finish:
        time.sleep(5)

    def __reset(self):
        print(f"\nin reset()\n")
        if len(self.actor_list) > 0:
            print(f"destruyendo actors_list[]")
            self.destroy_all_actors()

        self.collision_hist = []
        # self.actor_list = []

        # -- vehicle random pose
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.car_model, self.transform)
        # self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
        # self.vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)
        # print(f"\n{self.actor_list=}\n")

        # -- camera definition
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        # self.rgb_cam.set_attribute("image_size_x", f"{self.width_image}")
        self.rgb_cam.set_attribute("image_size_x", "640")
        # self.rgb_cam.set_attribute("image_size_y", f"{self.height_image}")
        self.rgb_cam.set_attribute("image_size_y", "480")
        self.rgb_cam.set_attribute("fov", "110")

        # -- camera attach
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor_camera = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle
        )
        self.sensor_camera.listen(lambda data: self.process_img(data))
        # self.sensor_camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))
        self.actor_list.append(self.sensor_camera)
        # print(f"\n{self.actor_list=}\n")

        time.sleep(10)

        # -- collision sensor
        # colsensor = self.blueprint_library.find("sensor.other.collision")
        # self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        # self.actor_list.append(self.colsensor)
        # self.colsensor.listen(lambda event: self.collision_data(event))

        # while self.rgb_cam is None:
        #    time.sleep(0.01)

        self.episode_start = time.time()
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.sensor_camera

    def _process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        # i2 = i.reshape((self.height_image, self.width_image, 4))
        i2 = i.reshape((480, 640, 4))
        # print(f"{i2=}")
        i3 = i2[:, :, :3]
        # if True:
        cv2.imshow("", i3)
        cv2.waitKey(1)
        # self.sensor_camera = i3
        return i3 / 255.0

    def process_img_(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((480, 640, 4))
        i3 = i2[:, :, :3]
        array2pil = im.fromarray(i3)
        array2pil.show()
        # array2pil_reduced = array2pil.convert(
        #    "P", palette=im.ADAPTIVE, colors=4
        # )
        # array2pil_reduced.show()
        # image = im.frombuffer(image.raw_data)
        # image.show()

    def process_img(self, image):
        # imgplot = plt.imshow(mpimg.imread(image.raw_data))
        # plt.ion()
        # plt.show()
        # plt.close()
        i = np.array(image.raw_data)
        i2 = i.reshape((480, 640, 4))
        i3 = i2[:, :, :3]
        plt.imshow(i3)
        # plt.plot(i3)
        # plt.show()
        # plt.pause(0.5)
        # plt.draw()

    #################################################################################################33
    def collision_data(self, event):
        self.collision_hist.append(event)

    def set_carlaclient_world_map(self, config):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)  # seconds
        # self.world = self.client.load_world(config['town'])
        # self.carla_map = self.world.get_map()
        self.world = self.client.get_world()

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = (
            0.05  # simulator takes 1/0.1 steps, i.e. 10 steps
        )
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)

    def set_fix_weather(self):
        """
        ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon,
        MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset,
        WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset,
        HardRainSunset.
        """
        # TODO: carla.WeatherParameters is a enum. Every time weather is changing
        # for i in enum(carla.WeatherParameters )
        #     weather.tick(speed_factor * elapsed_time)
        #     world.set_weather(weather.weather)

        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def set_arg_weather(self, dynamic=None):
        """
        setting weather through params in utils.get_args()
        """
        self.weather = self.world.get_weather()
        args = get_args()
        # apply presets
        apply_sun_presets(args, self.weather)
        apply_weather_presets(args, self.weather)
        # apply weather values individually
        apply_weather_values(args, self.weather)

        self.world.set_weather(self.weather)

    def set_street_lights(self):
        pass

    def set_car_lights(self):
        pass

    def get_snapshot(self):
        """TODO: get snapshots for a particular actor"""
        pass

    def destroy_all_actors(self):
        for actor in self.actor_list:
            actor.destroy()
        # self.actor_list = []
