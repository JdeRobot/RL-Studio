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

        # self.display_manager = None
        # self.vehicle = None
        # self.actor_list = []
        self.timer = CustomTimer()

        self.client = carla.Client(
            config["carla_server"],
            config["carla_client"],
        )
        self.client.set_timeout(5.0)
        print(f"\n maps in carla 0.9.13: {self.client.get_available_maps()}\n")

        self.world = self.client.load_world(config["town"])
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        if config["sync"]:
            self.traffic_manager.set_synchronous_mode(True)
        else:
            self.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(settings)

        # self.camera = None
        # self.vehicle = None
        # self.display = None
        # self.image = None

        ## -- display manager
        self.display_manager = DisplayManager(
            grid_size=[3, 4],
            window_size=[1500, 800],
        )

        self.car = None

    def reset(self):

        # print(f"=============== RESET ===================")
        self.collision_hist = []
        self.actor_list = []
        # self.display_manager.actor_list = []

        ## ---  Car
        self.setup_car_random_pose()
        ## --- Sensor collision
        self.setup_col_sensor()

        ## --- Cameras
        self.camera_spectator = SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=-5, z=2.8), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 0],
        )
        self.camera_spectator_segmentated = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCamera",
            carla.Transform(carla.Location(x=-5, z=2.8), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[1, 0],
        )
        self.sergio_camera_spectator = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCameraSergio",
            carla.Transform(carla.Location(x=-5, z=2.8), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[2, 0],
        )
        self.front_camera = SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 1],
        )

        self.front_camera_segmentated = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCamera",
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[1, 1],
        )

        self.sergio_front_camera = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCameraSergio",
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[2, 1],
        )
        self.front_camera_mas_baja = SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=2, z=0.5), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 2],
        )

        self.front_camera_mas_baja_segmentated = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCamera",
            carla.Transform(carla.Location(x=2, z=0.5), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[1, 2],
        )

        # self.sergio_front_camera_mas_baja = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "SemanticCameraSergio",
        #    carla.Transform(carla.Location(x=2, z=0.5), carla.Rotation(yaw=+0)),
        #    self.car,
        #    {},
        #    display_pos=[2, 2],
        # )

        # self.front_camera_mas_baja_bev = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "BirdEyeView",
        #    carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[2, 2],
        # )

        self.front_camera_1_5 = SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 3],
        )

        self.front_camera_1_5_segmentated = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCamera",
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[1, 3],
        )

        self.sergio_front_camera_1_5 = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCameraSergio",
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+0)),
            self.car,
            {},
            display_pos=[2, 3],
        )

        time.sleep(1)
        self.episode_start = time.time()
        self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        #### VAMOs a enganar al step
        # stados = random.randint(0, 16)
        # stados = [stados]
        # print(f"stados = {stados}")

        ## -- states
        states, _, _ = self.calculate_states(
            self.front_camera.front_camera,
            self.sergio_front_camera_1_5.front_camera_sergio_segmentation,
        )

        return states

    ####################################################
    ####################################################
    def calculate_states(self, image, red_mask):

        # print(
        #    f"ORIGINAL IMAGE SIZE: height = {red_mask.shape[0]}, width = {red_mask.shape[1]}, center = {red_mask.shape[1]//2}"
        # )

        ## first, we cut the upper image
        height = red_mask.shape[0]
        # width = red_mask.shape[1]
        # center_image = width // 2
        image_middle_line = (height) // 2
        img_sliced = red_mask[image_middle_line:]
        ## calculating new image measurements
        height = img_sliced.shape[0]
        width = img_sliced.shape[1]
        center_image = width // 2

        ## -- convert to GRAY
        gray_mask = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)

        ## --  aplicamos mascara para convertir a BLANCOS Y NEGROS
        _, white_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

        # self.show_image("image", image, 1)
        self.show_image("red mask", cv2.cvtColor(red_mask, cv2.COLOR_BGR2RGB), 1)
        # self.show_image("gray mask", gray_mask, 1)
        self.show_image("white mask", white_mask, 1)

        # print(f"{height = }, {width = }, {center_image = }")
        # print(f"{self.x_row = }")
        ## get total lines in every line point
        lines = [white_mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        ## As we drive in right lane, we get from right to left
        lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from right lane to center
        inv_index_right = [
            np.argmax(lines_inversed[x]) for x, _ in enumerate(lines_inversed)
        ]
        # print(f"{inv_index_right = }\n")
        index_right = [
            width - inv_index_right[x] for x, _ in enumerate(inv_index_right)
        ]
        # print(f"{index_right = }\n")
        distance_to_center = [
            width - inv_index_right[x] - center_image
            for x, _ in enumerate(inv_index_right)
        ]
        # print(f"{distance_to_center = }\n")
        ## normalized distances
        distance_to_center_normalized = [
            abs(float((center_image - index_right[i]) / center_image))
            for i, _ in enumerate(index_right)
        ]
        # print(f"distance_to_center_normalized = {distance_to_center_normalized}")

        # TODO: mostrar la imagen con las 4 lineas de las filas y la posicion del centro

        ## calculating states
        states = []
        states2 = []
        for _, x in enumerate(index_right):
            states.append(int(x / 40))
            states2.append(int(x / (width / self.num_regions)))
        # print(f"states:{states}\n")
        # print(f"states2:{states2}\n")

        return states, distance_to_center, distance_to_center_normalized

    def show_image(self, name, img, waitkey):
        window_name = f"{name}"
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    def setup_car_random_pose(self):
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(car_bp, location)
        while self.car is None:
            self.car = self.world.try_spawn_actor(car_bp, location)
        self.actor_list.append(self.car)
        time.sleep(1)

    def setup_col_sensor(self):
        colsensor = self.world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.car
        )
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

    def collision_data(self, event):
        self.collision_hist.append(event)

    def destroy_all_actors(self):
        for actor in self.actor_list[::-1]:
            # for actor in self.actor_list:
            actor.destroy()
        # print(f"\nin self.destroy_all_actors(), actor : {actor}\n")

        # self.actor_list = []
        # .client.apply_batch(
        #    [carla.command.DestroyActor(x) for x in self.actor_list[::-1]]
        # )

    #################################################################################
    def step(self, action):
        # print(f"=============== STEP ===================")

        ### -------- send action
        params = self.control(action)
        print(f"params = {params}")

        # params["pos"] = 270
        # center = 270
        # stados = random.randint(0, 4)
        # stados = [stados]
        # print(f"stados = {stados}")

        ## -- states
        (
            states,
            distance_to_center,
            distance_to_center_normalized,
        ) = self.calculate_states(
            self.front_camera.front_camera,
            self.sergio_front_camera_1_5.front_camera_sergio_segmentation,
        )
        # print(f"states:{states}\n")

        ## -------- Rewards
        reward, done = self.rewards_followlane_center_v_w()

        return states, reward, done, {}

    def control(self, action):

        steering_angle = 0
        if action == 0:
            self.car.apply_control(carla.VehicleControl(throttle=1, steer=-0.2))
            steering_angle = 0.2
        elif action == 1:
            self.car.apply_control(carla.VehicleControl(throttle=1, steer=0.0))
            steering_angle = 0
        elif action == 2:
            self.car.apply_control(carla.VehicleControl(throttle=1, steer=0.2))
            steering_angle = 0.2
        # elif action == 3:
        #    self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.4))
        #    steering_angle = 0.4
        # elif action == 4:
        #    self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.4))
        #    steering_angle = 0.4
        else:
            print("error in action")
            pass
        params = {}

        v = self.car.get_velocity()
        params["velocity"] = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        w = self.car.get_angular_velocity()
        params["steering_angle"] = steering_angle

        return params

    def rewards_followlane_center_v_w(self):
        center = 0.3
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


###########################################################################################################
############################################################################################################
#############################################################################################################
class OLD_FollowLaneQlearnStaticWeatherNoTraffic(FollowLaneEnv):
    def __init__(self, **config):

        print(f"in FollowLaneQlearnStaticWeatherNoTraffic -> launching FollowLaneEnv\n")
        ###### init F1env
        FollowLaneEnv.__init__(self, **config)
        ###### init class variables
        print(f"leaving FollowLaneEnv\n ")
        print(f"launching FollowLaneCarlaConfig\n ")
        FollowLaneCarlaConfig.__init__(self, **config)

        self.client = carla.Client(
            config["carla_server"],
            config["carla_client"],
        )
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05

        if config["sync"]:
            settings.synchronous_mode = True
        else:
            print(f"activo async")
            settings.synchronous_mode = False

        self.world.apply_settings(settings)

        self.camera = None
        self.vehicle = None
        self.display = None
        self.image = None

        self.timer = CustomTimer()
        self.time_processing = 0.0
        self.tics_processing = 0

        self.image_dict = {}

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        # car_bp = self.blueprint_library.filter("vehicle.*")[0]

        ## --- Pygame Display
        pygame.init()
        pygame.font.init()
        self.gameDisplay = pygame.display.set_mode(
            (1500, 800), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.clock = pygame.time.Clock()
        self.surface = None

    def __del__(self):
        print("__del__ called")

    def reset(self):
        """
        reset for
        - Algorithm: Q-learn
        - State: Simplified perception
        - tasks: FollowLane
        """

        self.collision_hist = []
        self.actor_list = []
        self.image_dict = {}

        ## ---  CAR
        self.setup_car()
        """
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        while self.vehicle is None:
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        time.sleep(2.0)
        """

        ## --- CAMERA FRONT
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
        ## In case of problem in getting rid off sensors ####
        ## We need to pass the lambda a weak reference to self to avoid circular
        ## reference.
        # weak_self = weakref.ref(self)
        # self.sensor_front_camera.listen(
        #    lambda data: FollowLaneQlearnStaticWeatherNoTraffic._weak_process_image(
        #        weak_self, data
        #    )
        # )
        self.sensor_front_camera.listen(lambda data: self.process_image(data))
        while self.sensor_front_camera is None:
            time.sleep(0.01)
        self.actor_list.append(self.sensor_front_camera)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2.0)

        ## --- Collision Sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.vehicle
        )
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        #### VAMOs a enganar al step
        stados = random.randint(0, 16)
        stados = [stados]
        # print(f"stados = {stados}")
        return stados

        # return self.front_camera

    def setup_car(self):
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        while self.vehicle is None:
            self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        time.sleep(2.0)

    def process_image(self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        print(f"--- reading image ------------------")
        if not isinstance(image, carla.Image):
            raise ValueError("Argument must be a carla.Image")
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.image_dict["image"] = array

        # if self.display_manager.render_enabled():
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1
        if self.surface is not None:
            # offset = self.display_manager.get_display_offset([0, 0])
            self.gameDisplay.blit(self.surface, (640, 480))
        self.clock.tick(60)

        # print(f"self.image_dict = {self.image_dict}")
        # cv2.imshow("", array)
        # cv2.waitKey(1)
        # time.sleep(0.1)
        self.front_camera = array

    """
    def render(self):
        if self.surface is not None:
            offset = self.display_manager.get_display_offset([0, 0])
            self.display_manager.display.blit(self.surface, offset)
    """

    @staticmethod
    def _weak_process_image(weak_self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""

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

    def collision_data(self, event):
        self.collision_hist.append(event)

    def destroy_all_actors(self):
        for actor in self.actor_list[::-1]:
            actor.destroy()

    #################################################
    #################################################
    def step(self, action):

        # print(f"entramos en step()")
        ### -------- send action
        params = self.control(action)
        # print(f"params = {params}")

        ### -------- State get center lane
        # weak_self = weakref.ref(self)
        # self.sensor_front_camera.listen(
        #    lambda data: FollowLaneQlearnStaticWeatherNoTraffic.process_image_weak(
        #        weak_self, data
        #    )
        # )
        # self.sensor_front_camera.listen(lambda data: self.process_image(data))
        # time.sleep(2.0)
        # while self.sensor_front_camera is None:
        #    time.sleep(0.01)
        #    print(f"entro")

        # params["pos"] = 270
        center = 270
        stados = random.randint(0, 4)
        stados = [stados]
        # print(f"stados = {stados}")
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
        # elif action == 3:
        #    self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.4))
        #    steering_angle = 0.4
        # elif action == 4:
        #    self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.4))
        #    steering_angle = 0.4
        else:
            print("error en action")
            pass
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

        SensorManager(
            self.world,
            self.display_manager,
            "SemanticCameraSergio",
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[1, 2],
        )

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
        print(image)
        print("process_img_sergio")
        pass
