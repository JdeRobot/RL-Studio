from collections import Counter
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
    render_params,
)
from rl_studio.envs.carla.followlane.followlane_env import FollowLaneEnv
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils

from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
    CustomTimer,
)
import pygame
from rl_studio.envs.carla.utils.global_route_planner import (
    GlobalRoutePlanner,
)


class FollowLaneQlearnStaticWeatherNoTraffic(FollowLaneEnv):
    def __init__(self, **config):

        ###### init F1env
        FollowLaneEnv.__init__(self, **config)
        ###### init class variables
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

        """
        self.world = self.client.get_world()
        self.world = self.client.load_world(config["town"])
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        if config["sync"]:
            settings.synchronous_mode = True
        else:
            settings.synchronous_mode = False
        self.world.apply_settings(settings)

        """

        self.world = self.client.load_world(config["town"])
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.5
        if config["sync"]:
            self.traffic_manager.set_synchronous_mode(True)
        else:
            self.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(settings)

        """este no funciona
        self.world = self.client.load_world(config["town"])
        # self.original_settings = self.world.get_settings()
        # self.traffic_manager = self.client.get_trafficmanager(8000)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        if config["sync"]:
            settings.synchronous_mode = True
        else:
            settings.synchronous_mode = False
        self.world.apply_settings(settings)
        """

        # self.camera = None
        # self.vehicle = None
        # self.display = None
        # self.image = None

        ## -- display manager
        # self.display_manager = DisplayManager(
        #    grid_size=[2, 2],
        #    window_size=[900, 600],
        # )

        self.car = None

        self.perfect_distance_pixels = None
        self.perfect_distance_normalized = None
        # self._control = carla.VehicleControl()
        self.params = {}
        self.target_waypoint = None

        self.front_rgb_camera = None
        self.front_red_mask_camera = None
        self.front_camera_bev = None
        self.spectator = self.world.get_spectator()

    def reset(self):

        self.collision_hist = []
        self.actor_list = []

        ## ---  Car
        waypoints_town = self.world.get_map().generate_waypoints(5.0)
        init_waypoint = waypoints_town[self.waypoints_init + 1]

        if self.alternate_pose:
            self.setup_car_random_pose()
        elif self.waypoints_target is not None:
            # waypoints = self.get_waypoints()
            filtered_waypoints = self.draw_waypoints(
                waypoints_town,
                self.waypoints_init,
                self.waypoints_target,
                self.waypoints_lane_id,
                2000,
            )
            self.target_waypoint = filtered_waypoints[-1]
            # print(f"{self.target_waypoint=}")
            # print(f"{self.target_waypoint.transform.location.x=}")
            # print(f"{self.target_waypoint.transform.location.y=}")
            # print(f"{self.target_waypoint.road_id=}")
            # print(f"{self.target_waypoint.lane_id=}")
            # print(f"{self.target_waypoint.id=}")

            self.setup_car_fix_pose(init_waypoint)

        else:  # TODO: hacer en el caso que se quiera poner el target con .next()
            waypoints_lane = init_waypoint.next_until_lane_end(1000)
            waypoints_next = init_waypoint.next(1000)
            print(f"{init_waypoint.transform.location.x = }")
            print(f"{init_waypoint.transform.location.y = }")
            print(f"{init_waypoint.lane_id = }")
            print(f"{init_waypoint.road_id = }")
            print(f"{len(waypoints_lane) = }")
            print(f"{len(waypoints_next) = }")
            w_road = []
            w_lane = []
            for x in waypoints_next:
                w_road.append(x.road_id)
                w_lane.append(x.lane_id)

            counter_lanes = Counter(w_lane)
            counter_road = Counter(w_road)
            print(f"{counter_lanes = }")
            print(f"{counter_road = }")

            self.setup_car_fix_pose(init_waypoint)

        ## --- Sensor collision
        self.setup_col_sensor()

        ## --- Cameras
        self.setup_rgb_camera()
        # self.setup_semantic_camera()
        while self.front_rgb_camera is None:
            # print("1")
            time.sleep(0.01)

        self.setup_red_mask_camera()
        # self.setup_bev_camera()

        while self.front_red_mask_camera is None:
            time.sleep(0.01)

        # AutoCarlaUtils.show_images(
        #    "main", self.front_rgb_camera, self.front_red_mask_camera, 600, 5
        # )

        time.sleep(1)
        self.episode_start = time.time()
        self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # AutoCarlaUtils.show_image("image", self.front_camera_1_5.front_camera, 1, 0, 10)

        mask = self.preprocess_image(self.front_red_mask_camera)
        # mask = self.preprocess_image(
        #    self.front_camera_1_5_red_mask.front_camera_red_mask
        # )
        # AutoCarlaUtils.show_image(
        #    "mask", mask, 1, self.front_camera_1_5.front_camera.shape[1], 10
        # )

        print(
            f"{self.front_rgb_camera.shape[0] = }, {self.front_rgb_camera.shape[1] = }"
        )
        print(
            f"{self.front_red_mask_camera.shape[0] = }, {self.front_red_mask_camera.shape[1] = }"
        )
        print(f"{mask.shape[0] = }, {mask.shape[1] = }")

        AutoCarlaUtils.show_image(
            "RGB",
            self.front_rgb_camera,
            500,
            10,
        )
        AutoCarlaUtils.show_image(
            "red mask",
            self.front_red_mask_camera,
            1000,
            10,
        )

        # Wait for world to get the vehicle actor
        time.sleep(5)
        self.world.tick()

        world_snapshot = self.world.wait_for_tick()
        actor_snapshot = world_snapshot.find(self.car.id)
        # Set spectator at given transform (vehicle transform)
        self.spectator.set_transform(actor_snapshot.get_transform())

        ## -- states
        (
            states,
            distance_center,
            distance_to_centr_normalized,
        ) = self.calculate_states(mask)

        self.perfect_distance_pixels = distance_center
        self.perfect_distance_normalized = distance_to_centr_normalized

        return states

    ####################################################
    #
    # Reset Methods
    ####################################################

    def calculate_states(self, mask):
        """
        from right, search for red line. It could be weakness to a bad masking
        """
        width = mask.shape[1]
        center_image = width // 2
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        ## As we drive in right lane, we get from right to left
        lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from right lane to center
        inv_index_right = [
            np.argmax(lines_inversed[x]) for x, _ in enumerate(lines_inversed)
        ]
        index_right = [
            width - inv_index_right[x] for x, _ in enumerate(inv_index_right)
        ]
        distance_to_center = [
            width - inv_index_right[x] - center_image
            for x, _ in enumerate(inv_index_right)
        ]
        ## normalized distances NO ABS
        distance_to_center_normalized = [
            float((center_image - index_right[i]) / center_image)
            for i, _ in enumerate(index_right)
        ]
        pixels_in_state = mask.shape[1] / self.num_regions
        states = [int(value / pixels_in_state) for _, value in enumerate(index_right)]

        return states, distance_to_center, distance_to_center_normalized

    def preprocess_image(self, red_mask):
        ## first, we cut the upper image
        height = red_mask.shape[0]
        image_middle_line = (height) // 2
        img_sliced = red_mask[image_middle_line:]
        ## calculating new image measurements
        height = img_sliced.shape[0]
        ## -- convert to GRAY
        gray_mask = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        ## --  aplicamos mascara para convertir a BLANCOS Y NEGROS
        _, white_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

        return white_mask

    ########################################################
    #  waypoints, car pose, sensors
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

    def setup_car_fix_pose(self, init):
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
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

        self.car = self.world.spawn_actor(car_bp, location)
        while self.car is None:
            self.car = self.world.spawn_actor(car_bp, location)

        self.actor_list.append(self.car)
        time.sleep(1)

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

    ##################
    #
    # Cameras
    ##################

    def setup_rgb_camera(self):
        print("enter setup_rg_camera")
        rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
        rgb_cam.set_attribute("image_size_x", f"{self.width}")
        rgb_cam.set_attribute("image_size_y", f"{self.height}")
        rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        camera_rgb = self.world.spawn_actor(rgb_cam, transform, attach_to=self.car)
        self.actor_list.append(camera_rgb)
        print(f"{len(self.actor_list) = }")
        camera_rgb.listen(self.save_rgb_image)

    def save_rgb_image(self, data: carla.Image):
        """
        @param data: pure carla.Image
        """
        print(f"entro awui")
        """Convert a CARLA raw image to a BGRA numpy array."""
        if not isinstance(data, carla.Image):
            raise ValueError("Argument must be a carla.Image")
        image = np.array(data.raw_data)
        image = image.reshape((480, 640, 4))
        image = image[:, :, :3]
        # self._data_dict["image"] = image3
        cv2.imshow("", image)
        cv2.waitKey(1)
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
        cv2.imshow("", array)
        cv2.waitKey(1)
        self.front_rgb_camera = array

    def setup_red_mask_camera(self):
        self.red_mask_cam = self.world.get_blueprint_library().find(
            "sensor.camera.semantic_segmentation"
        )
        self.red_mask_cam.set_attribute("image_size_x", f"{self.width}")
        self.red_mask_cam.set_attribute("image_size_y", f"{self.height}")

        transform = carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00))
        self.camera_red_mask = self.world.spawn_actor(
            self.red_mask_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.camera_red_mask)
        self.camera_red_mask.listen(
            lambda data: self.save_red_mask_semantic_image(data)
        )

    def save_red_mask_semantic_image(self, image):
        # t_start = self.timer.time()
        print(f"en red_mask calbback")
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        hsv_nemo = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        light_sidewalk = (151, 217, 243)
        dark_sidewalk = (153, 219, 245)

        light_pavement = (149, 127, 127)
        dark_pavement = (151, 129, 129)

        mask_sidewalk = cv2.inRange(hsv_nemo, light_sidewalk, dark_sidewalk)
        result_sidewalk = cv2.bitwise_and(array, array, mask=mask_sidewalk)

        mask_pavement = cv2.inRange(hsv_nemo, light_pavement, dark_pavement)
        result_pavement = cv2.bitwise_and(array, array, mask=mask_pavement)

        # Adjust according to your adjacency requirement.
        kernel = np.ones((3, 3), dtype=np.uint8)

        # Dilating masks to expand boundary.
        color1_mask = cv2.dilate(mask_sidewalk, kernel, iterations=1)
        color2_mask = cv2.dilate(mask_pavement, kernel, iterations=1)

        # Required points now will have both color's mask val as 255.
        common = cv2.bitwise_and(color1_mask, color2_mask)
        SOME_THRESHOLD = 10

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

        self.front_red_mask_camera = red_line_mask

    ##################
    #
    #
    ##################

    def collision_data(self, event):
        self.collision_hist.append(event)
        actor_we_collide_against = event.other_actor
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        print(f"you crashed with {actor_we_collide_against.type_id}")
        # self.actor_list.append(actor_we_collide_against)

    def destroy_all_actors(self):
        print("\ndestroying %d actors" % len(self.actor_list))
        for actor in self.actor_list[::-1]:
            # for actor in self.actor_list:
            print(f"destroying actor : {actor}\n")
            actor.destroy()
        print(f"All actors destroyed")
        # for collisions in self.collision_hist:
        # for actor in self.actor_list:
        # collisions.destroy()
        # print(f"\nin self.destroy_all_actors(), actor : {actor}\n")

        # self.actor_list = []
        # .client.apply_batch(
        #    [carla.command.DestroyActor(x) for x in self.actor_list[::-1]]
        # )

    #################################################
    #
    # Step
    #################################################

    def step(self, action):
        ### -------- send action
        self.control(action)
        # print(f"params = {params}")

        # params["pos"] = 270
        # center = 270
        # stados = random.randint(0, 4)
        # stados = [stados]
        # print(f"stados = {stados}")

        AutoCarlaUtils.show_images(
            "main", self.front_rgb_camera, self.front_red_mask_camera, 300, 500
        )
        ## -- states
        mask = self.preprocess_image(
            self.front_camera_1_5_red_mask.front_camera_red_mask
        )

        (
            states,
            distance_to_center,
            distance_to_center_normalized,
        ) = self.calculate_states(mask)

        # AutoCarlaUtils.show_image("mask", mask, 1, 500, 10)
        AutoCarlaUtils.show_image_with_centrals(
            "mask",
            mask,
            1,
            distance_to_center,
            distance_to_center_normalized,
            self.x_row,
            600,
            10,
        )
        # print(f"states:{states}\n")
        AutoCarlaUtils.show_image_with_centrals(
            "image",
            self.front_rgb_camera.front_camera[mask.shape[0] :],
            1,
            distance_to_center,
            distance_to_center_normalized,
            self.x_row,
            self.front_camera_1_5.front_camera.shape[1] + 600,
            10,
        )

        AutoCarlaUtils.show_images_tile(
            "imagesss",
            # [self.front_rgb_camera, self.front_red_mask_camera, mask],
            [self.front_rgb_camera, self.front_red_mask_camera],
            500,
            500,
        )
        ## ------ calculate distance error and states
        # print(f"{self.perfect_distance_normalized =}")
        error = [
            abs(
                self.perfect_distance_normalized[index]
                - distance_to_center_normalized[index]
            )
            for index, value in enumerate(self.x_row)
        ]
        counter_states = Counter(states)
        states_16 = counter_states.get(16)

        ## -------- Ending Step()...
        done = False
        ## -------- Rewards
        reward, done = self.autocarlarewards.rewards_easy(error, self.params)
        # reward, done = self.rewards_followlane_error_center(
        #    distance_to_center_normalized, self.rewards
        # )

        ## -------- ... or Finish by...
        if states_16 is not None and (states_16 >= len(states)):  # not red right line
            print(f"no red line detected")
            done = True
        if len(self.collision_hist) > 0:  # crash you, baby
            done = True
            print(f"crash")

        is_finish, dist = AutoCarlaUtils.finish_target(
            self.params["location"],
            self.target_waypoint,
            self.max_target_waypoint_distance,
        )
        if is_finish:
            print(f"Finish!!!!")
            done = True

        render_params(
            action=action,
            speed_kmh=self.params["speed"],
            # steering_angle=self.params["steering_angle"],
            Steer=self.params["Steer"],
            location=self.params["location"],
            Throttle=self.params["Throttle"],
            Brake=self.params["Brake"],
            height=self.params["height"],
            _="------------------------",
            states=states,
            distance_to_center=distance_to_center,
            distance_to_center_normalized=distance_to_center_normalized,
            reward=reward,
            error=error,
            done=done,
            states_16=states_16,
            self_collision_hist=self.collision_hist,
            distance_to_finish=dist,
        )
        """
        print_messages(
            "in step()",
            height=mask.shape[0],
            width=mask.shape[1],
            action=action,
            velocity=self.params["speed"],
            # steering_angle=self.params["steering_angle"],
            Steer=self.params["Steer"],
            location=self.params["location"],
            Throttle=self.params["Throttle"],
            Brake=self.params["Brake"],
            _="------------------------",
            states=states,
            distance_to_center=distance_to_center,
            distance_to_center_normalized=distance_to_center_normalized,
            self_perfect_distance_pixels=self.perfect_distance_pixels,
            self_perfect_distance_normalized=self.perfect_distance_normalized,
            error=error,
            done=done,
            reward=reward,
            states_16=states_16,
            self_collision_hist=self.collision_hist,
        )
        """
        return states, reward, done, {}

    def clip_throttle(self, throttle, curr_speed, target_speed):
        """
        limits throttle in function of current speed and target speed
        """
        clip_throttle = np.clip(throttle - 0.1 * (curr_speed - target_speed), 0.25, 1.0)
        return clip_throttle

    def control(self, action):

        steering_angle = 0
        if action == 0:
            self.car.apply_control(
                carla.VehicleControl(throttle=0.3, steer=0.1)
            )  # jugamos con -0.01
            # self._control.throttle = min(self._control.throttle + 0.01, 1.00)
            # self._control.steer = -0.02
            steering_angle = -0.01
        elif action == 1:
            self.car.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0))
            # self._control.throttle = min(self._control.throttle + 0.01, 1.00)
            # self._control.steer = 0.0
            steering_angle = 0
        elif action == 2:
            self.car.apply_control(
                carla.VehicleControl(throttle=0.3, steer=0.1)
            )  # jigamos con 0.01 par ala recta
            # self._control.throttle = min(self._control.throttle + 0.01, 1.0)
            # self._control.steer = 0.02
            steering_angle = 0.01

        # self.car.apply_control(self._control)

        t = self.car.get_transform()
        v = self.car.get_velocity()
        c = self.car.get_control()
        w = self.car.get_angular_velocity()
        self.params["speed"] = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.params["steering_angle"] = w
        # print(f"{self.params['steering_angle'].x = }")
        # print(f"{self.params['steering_angle'].y = }")
        # print(f"{self.params['steering_angle'].z = }")

        self.params["Steering_angle"] = steering_angle
        self.params["Steer"] = c.steer
        self.params["location"] = (t.location.x, t.location.y)
        self.params["Throttle"] = c.throttle
        self.params["Brake"] = c.brake
        self.params["height"] = t.location.z

    ########################################################
    #  utils
    #
    ########################################################

    def __del__(self):
        print("__del__ called")

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
