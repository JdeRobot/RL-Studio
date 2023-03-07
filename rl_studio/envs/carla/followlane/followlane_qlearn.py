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
            grid_size=[2, 3],
            window_size=[1500, 800],
        )

        self.car = None

        self.perfect_distance_pixels = None
        self.perfect_distance_normalized = None
        # self._control = carla.VehicleControl()
        self.params = {}

    def reset(self):

        self.collision_hist = []
        self.actor_list = []
        spectator = self.world.get_spectator()

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

        self.front_camera_1_5_bev = SensorManager(
            self.world,
            self.display_manager,
            "BirdEyeView",
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[1, 1],
            client=self.client,
        )

        self.front_camera_1_5 = SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 0],
        )

        self.front_camera_1_5_segmentated = SensorManager(
            self.world,
            self.display_manager,
            "SemanticCamera",
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 1],
        )

        self.front_camera_1_5_red_mask = SensorManager(
            self.world,
            self.display_manager,
            "RedMask",
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+0)),
            self.car,
            {},
            display_pos=[0, 2],
        )

        time.sleep(1)
        self.episode_start = time.time()
        self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # AutoCarlaUtils.show_image("image", self.front_camera_1_5.front_camera, 1, 0, 10)

        mask = self.preprocess_image(
            self.front_camera_1_5_red_mask.front_camera_red_mask
        )
        # AutoCarlaUtils.show_image(
        #    "mask", mask, 1, self.front_camera_1_5.front_camera.shape[1], 10
        # )

        # Wait for world to get the vehicle actor
        self.world.tick()

        world_snapshot = self.world.wait_for_tick()
        actor_snapshot = world_snapshot.find(self.car.id)
        # Set spectator at given transform (vehicle transform)
        spectator.set_transform(actor_snapshot.get_transform())

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

    def draw_waypoints(self, spawn_points, init, target, lane_id, life_time):
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
            self.front_camera_1_5.front_camera[mask.shape[0] :],
            1,
            distance_to_center,
            distance_to_center_normalized,
            self.x_row,
            self.front_camera_1_5.front_camera.shape[1] + 600,
            10,
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

        done = False

        ## -------- Rewards
        reward, done = self.rewards_easy(error, self.params)
        # reward, done = self.rewards_followlane_error_center(
        #    distance_to_center_normalized, self.rewards
        # )

        ## -------- Finish by...
        if states_16 is not None and (
            states_16 > (len(states) // 2)
        ):  # not red right line
            print(f"no red line detected")
            done = True
        if len(self.collision_hist) > 0:  # crash you, baby
            done = True
            print(f"crash")

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
            done=done,
        )
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

        return states, reward, done, {}

    def control(self, action):

        steering_angle = 0
        if action == 0:
            self.car.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.1))
            # self._control.throttle = min(self._control.throttle + 0.01, 1.00)
            # self._control.steer = -0.02
            steering_angle = -0.02
        elif action == 1:
            self.car.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0))
            # self._control.throttle = min(self._control.throttle + 0.01, 1.00)
            # self._control.steer = 0.0
            steering_angle = 0
        elif action == 2:
            self.car.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.1))
            # self._control.throttle = min(self._control.throttle + 0.01, 1.0)
            # self._control.steer = 0.02
            steering_angle = 0.02

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

    def rewards_followlane_dist_v_angle(self, error, params):
        # rewards = []
        # for i,_ in enumerate(error):
        #    if (error[i] < 0.2):
        #        rewards.append(10)
        #    elif (0.2 <= error[i] < 0.4):
        #        rewards.append(2)
        #    elif (0.4 <= error[i] < 0.9):
        #        rewards.append(1)
        #    else:
        #        rewards.append(0)
        rewards = [0.1 / error[i] for i, _ in enumerate(error)]
        function_reward = sum(rewards) / len(rewards)
        function_reward += math.log10(params["velocity"])
        function_reward -= 1 / (math.exp(params["steering_angle"]))

        return function_reward

    def rewards_followline_center(self, center, rewards):
        """
        original for Following Line
        """
        done = False
        if center > 0.9:
            done = True
            reward = rewards["penal"]
        elif 0 <= center <= 0.2:
            reward = rewards["from_10"]
        elif 0.2 < center <= 0.4:
            reward = rewards["from_02"]
        else:
            reward = rewards["from_01"]

        return reward, done

    def rewards_followlane_error_center(self, error, rewards):
        """
        original for Following Line
        """
        done = False
        if error > 0.9:
            done = True
            reward = rewards["penal"]
        elif 0 <= error <= 0.2:
            reward = rewards["from_10"]
        elif 0.2 < error <= 0.4:
            reward = rewards["from_02"]
        else:
            reward = rewards["from_01"]

        return reward, done

    def rewards_easy(self, error, params):
        rewards = []
        done = False
        for i, _ in enumerate(error):
            if error[i] < 0.2:
                rewards.append(10)
            elif 0.2 <= error[i] < 0.4:
                rewards.append(2)
            elif 0.4 <= error[i] < 0.9:
                rewards.append(1)
            else:
                rewards.append(-100)
                done = True

        function_reward = sum(rewards) / len(rewards)

        # TODO: remove next comments
        # function_reward += params["velocity"] * 0.5
        # function_reward -= params["steering_angle"] * 1.02

        return function_reward, done

    def rewards_followlane_center_v_w(self):
        """esta sin terminar"""
        center = 0
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
