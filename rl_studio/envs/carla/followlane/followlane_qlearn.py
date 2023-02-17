import math
import time
import carla
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np
import random
from datetime import datetime, timedelta
import weakref
import rospy
from sensor_msgs.msg import Image

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.carla.followlane.followlane_env import FollowLaneEnv
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig

from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
)
import pygame
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    CustomTimer,
)


class FollowLaneQlearnStaticWeatherNoTraffic(FollowLaneEnv):
    def __init__(self, **config):

        print(f"in FollowLaneQlearnStaticWeatherNoTraffic -> launching FollowLaneEnv\n")
        ###### init F1env
        FollowLaneEnv.__init__(self, **config)
        ###### init class variables
        print(f"leaving FollowLaneEnv\n ")
        print(f"launching FollowLaneCarlaConfig\n ")
        FollowLaneCarlaConfig.__init__(self, **config)

        # print(f"config = {config}")
        # ----------------------------
        # self.bsc = config["bsc"]
        # self.world = config["world"]
        # self.camera_rgb_front = config["camera_rgb_front"]
        # self.display_manager = config["display_manager"]

        self.client = carla.Client(
            config["carla_server"],
            config["carla_client"],
        )
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # set syncronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.camera = None
        self.vehicle = None
        self.display = None
        self.image = None

        # self.display_manager = DisplayManager(
        #    grid_size=[2, 3],
        #    window_size=[1500, 800],
        # )

        self.image_dict = {}
        # self.display_manager.add_sensor(self)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        """
        reset for
        - Algorithm: Q-learn
        - State: Simplified perception
        - tasks: FollowLane
        """

        # self.client.apply_batch(
        #    [carla.command.DestroyActor(x) for x in self.actor_list]
        # )
        # if len(self.actor_list) > 0:
        #    self.destroy_all_actors()
        #    print(f"entro reset destroy actors")
        # self.collision_hist = []
        # self.actor_list = []
        # time.sleep(1)
        print(f"entro reset()")
        self.collision_hist = []
        self.actor_list = []
        self.image_dict = {}
        print(f"len(image_dict) antes de leer imagen= {len(self.image_dict)}")
        ## -----------------------------------------------  COCHE
        # car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        while self.vehicle is None:
            print(f"entro here {datetime.now()}")
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        time.sleep(2.0)

        ## ----------------------------------------------- CAMERA FRONT
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", "640")
        self.rgb_cam.set_attribute("image_size_y", "480")
        self.rgb_cam.set_attribute("fov", "110")
        transform = carla.Transform(
            carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=+00)
        )
        self.sensor_front_camera = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle
        )
        self.actor_list.append(self.sensor_front_camera)

        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor_front_camera.listen(
            lambda data: FollowLaneQlearnStaticWeatherNoTraffic.process_image_weak(
                weak_self, data
            )
        )
        # self.front_camera.listen(lambda data: self.process_image(data))
        # self.front_camera.listen(self.process_img)
        # self.front_camera.listen(self.pruebaparaescuchar)
        # self.front_camera.listen(lambda data: self.pruebaparaescuchar(data))

        time.sleep(2.0)
        while self.sensor_front_camera is None:
            time.sleep(0.01)
            print(f"entro")

        # --------------- actuator
        self.vehicle.set_autopilot(True)
        time.sleep(1)

        for actor in self.actor_list:
            print(f"in reset - actor in self.actor_list: {actor} \n")

        print(f"len(image_dict) = {len(self.image_dict)}")

        #### VAMOs a enganar al step
        stados = random.randint(0, 4)
        stados = [stados]
        print(f"stados = {stados}")
        return stados

        # return self.front_camera

    def pruebaparaescuchar(self, data):
        print("------------------------entre por fin")

    def process_image(self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        if not isinstance(image, carla.Image):
            raise ValueError("Argument must be a carla.Image")
        image = np.array(image.raw_data)
        image2 = image.reshape((480, 640, 4))
        image3 = image2[:, :, :3]
        self.image_dict["image"] = image3
        # print(f"self.image_dict = {self.image_dict}")
        cv2.imshow("", image3)
        cv2.waitKey(1)
        time.sleep(0.1)
        print(f"holaaaaaaaaaaaaaa-----------------------------------")

    @staticmethod
    def process_image_weak(weak_self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        print(f"holaaaaaaaaaaaaaa-----------------------------------")

        self = weak_self()
        if not self:
            return

        if not isinstance(image, carla.Image):
            raise ValueError("Argument must be a carla.Image")

        image = np.array(image.raw_data)
        image2 = image.reshape((480, 640, 4))
        image3 = image2[:, :, :3]
        self.image_dict["image"] = image3
        # print(f"self.image_dict = {self.image_dict}")
        cv2.imshow("", image3)
        cv2.waitKey(1)
        time.sleep(0.1)
        self.front_camera = image3

    def destroy_all_actors(self):
        # for actor in self.actor_list[::-1]:
        for actor in self.actor_list:
            actor.destroy()
            print(f"\nin self.destroy_all_actors(), actor : {actor}\n")

        self.actor_list = []

    #################################################
    #################################################
    def step(self, action, step):

        print(f"entramos en step()")
        ### -------- send action
        params = self.control(action)
        print(f"params = {params}")

        ### -------- State get center lane
        weak_self = weakref.ref(self)
        self.sensor_front_camera.listen(
            lambda data: FollowLaneQlearnStaticWeatherNoTraffic.process_image_weak(
                weak_self, data
            )
        )
        params["pos"] = 270
        center = 270
        stados = random.randint(0, 4)
        stados = [stados]
        print(f"stados = {stados}")
        ## -------- Rewards
        reward, done = self.rewards_followlane_centerline(center)

        return stados, reward, done, {}

    def control(self, action):

        steering_angle = 0
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.32, steer=-0.2))
            steering_angle = 0.2
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0))
            steering_angle = 0
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.32, steer=0.2))
            steering_angle = 0.2
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.4))
            steering_angle = 0.4
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.4))
            steering_angle = 0.4

        params = {}

        v = self.vehicle.get_velocity()
        params["velocity"] = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        w = self.vehicle.get_angular_velocity()
        params["steering_angle"] = steering_angle

        return params

    def rewards_followlane_centerline(self, center):
        """
        works perfectly
        rewards in function of center of Line
        """
        done = False
        if 0.65 >= center > 0.25:
            reward = 10
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = 2
        elif 0 >= center > -0.9:
            reward = 1
        else:
            reward = -100
            done = True

        return reward, done

    ############################################################################################################

    def AAAAA_reset(self):
        """
        reset for
        - Algorithm: Q-learn
        - State: Simplified perception
        - tasks: FollowLane
        """
        if len(self.display_manager.get_sensor_list()) > 0:
            print(f"destruyendo sensors_list[]")
            print(
                f"len self.display_manager.sensor_list = {len(self.display_manager.sensor_list)}"
            )
            self.display_manager.destroy()

        # if len(self.actor_list) > 0:
        #    print(f"destruyendo actors_list[]")
        #    for actor in self.actor_list:
        #        actor.destroy()

        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.actor_list]
        )

        self.collision_hist = []
        self.actor_list = []
        print(f"----1. in reset() len(self.actor_list) = {len(self.actor_list)}")
        # self.display_manager.actor_list = []
        time.sleep(1)

        ## -----------------------------------------------  COCHE
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(car_bp, location)
        while self.car is None:
            # print(f"entro here {datetime.now()}")
            self.car = self.world.try_spawn_actor(car_bp, location)
        self.actor_list.append(self.car)
        ## ----------------------------------------------- CAMERA FRONT
        self.rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"640")
        self.rgb_cam.set_attribute("image_size_y", f"480")
        self.rgb_cam.set_attribute("fov", f"110")
        transform = carla.Transform(
            carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=+00)
        )
        self.front_camera = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.front_camera)

        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        # weak_self = weakref.ref(self)
        # self.front_camera.listen(
        #    lambda data: FollowLaneQlearnStaticWeatherNoTraffic.process_img(
        #        weak_self, data
        #    )
        # )
        self.front_camera.listen(lambda data: self.process_img(data))
        while self.front_camera is None:
            time.sleep(0.01)
        ## ----------------------------------------------- CAMERA FRONT
        camera = SensorManager(
            # SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=-4, z=2.4), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 0],
        )
        # self.actor_list.append(camera.actor_list[0])
        # --------------- actuator
        self.car.set_autopilot(True)

        for actor in self.actor_list:
            print(f"in reset - actor in self.actor_list: {actor} \n")

        for actor in self.display_manager.sensor_list:
            print(f"in reset - actor in self.display_manager.sensor_list: {actor} \n")

        # for actor in sensor_manager.actor_list:
        #    print(f"in reset - actor: {actor} \n")
        # for actor in self.display_manager.actor_list:
        #    print(f"in reset - actor in self.display_manager.actor_list: {actor}\n")

        return self.front_camera

    def _____reset(self):
        """
        esta funcionando pero no mata los sensores de SensorManager
        reset for
        - Algorithm: Q-learn
        - State: Simplified perception
        - tasks: FollowLane
        """

        print(f"\nin reset()\n")
        # if len(self.bsc.actor_list) > 0:
        #    print(f"destruyendo actors_list[]")
        #    for actor in self.bsc.actor_list:
        #        actor.destroy()
        # sensor_manager = SensorManager()

        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.actor_list]
        )
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.display_manager.actor_list]
        )
        print(
            f"----in reset() len(self.actor_list) = {len(self.actor_list)} y len(self.display_manager.actor_list) = {len(self.display_manager.actor_list)}"
        )
        self.collision_hist = []
        self.actor_list = []
        self.display_manager.actor_list = []
        time.sleep(2)

        ## -----------------------------------------------  COCHE
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(car_bp, location)
        while self.car is None:
            # print(f"entro here {datetime.now()}")
            self.car = self.world.try_spawn_actor(car_bp, location)
        self.actor_list.append(self.car)

        ## ----------------------------------------------- CAMERA FRONT
        self.rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"640")
        self.rgb_cam.set_attribute("image_size_y", f"480")
        self.rgb_cam.set_attribute("fov", f"110")
        transform = carla.Transform(
            carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=+00)
        )
        self.front_camera = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.front_camera)

        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        # weak_self = weakref.ref(self)
        # self.front_camera.listen(
        #    lambda data: FollowLaneQlearnStaticWeatherNoTraffic.process_img(
        #        weak_self, data
        #    )
        # )
        self.front_camera.listen(lambda data: self.process_img(data))
        while self.front_camera is None:
            time.sleep(0.01)

        ## ----------------------------------------------- CAMERA SERGIO
        self.sergio_camera = self.world.get_blueprint_library().find(
            "sensor.camera.rgb"
        )
        self.sergio_camera.set_attribute("image_size_x", f"640")
        self.sergio_camera.set_attribute("image_size_y", f"480")
        self.sergio_camera.set_attribute("fov", f"110")
        transformsergiocamera = carla.Transform(
            carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=+00)
        )
        self.front_camera_sergio = self.world.spawn_actor(
            self.sergio_camera, transformsergiocamera, attach_to=self.car
        )
        self.actor_list.append(self.front_camera_sergio)

        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self_sergio = weakref.ref(self)
        self.front_camera_sergio.listen(
            lambda data: FollowLaneQlearnStaticWeatherNoTraffic.process_img_sergio(
                weak_self_sergio, data
            )
        )

        while self.front_camera is None:
            time.sleep(0.01)

        # --------------- actuator
        self.car.set_autopilot(True)

        # self.display_manager.add_sensor(self.front_camera)
        # self.display_manager.render()

        # ---------------------------------- SHOW UP

        ### TODO: si se bloquea despues de muchos epochs, debe ser porque no esta eliminando estos sensores
        SensorManager(
            # SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=-4, z=2.4), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 0],
        )
        # self.actor_list.append(self.camera.actor_list[0])

        for actor in self.actor_list:
            print(f"in reset - actor in self.actor_list: {actor} \n")
        # for actor in sensor_manager.actor_list:
        #    print(f"in reset - actor: {actor} \n")
        for actor in self.display_manager.actor_list:
            print(f"in reset - actor in self.display_manager.actor_list: {actor}\n")

        return self.front_camera

    ####################################################
    ####################################################

    """
    def process_img(self, image):
        print(
            f"\n\nholaaaaaaaaaaaaaa-----------------------------------------------------------\n\n"
        )
    """
    """
    @staticmethod
    def process_img(weak_self, image):
        self = weak_self
        if not self:
            return

        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((480, 640, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)
        self.front_camera = i3
    """

    @staticmethod
    def process_img_sergio(weak_self, image):
        """
        esta es la funcion callback que procesa la imagen y la segmenta
        """
        pass
