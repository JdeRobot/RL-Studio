import os
import weakref
from collections import Counter
import math
import time
import carla
import random
import cv2
import torch
from numpy import random
import numpy as np
from rl_studio.envs.carla.utils.YOLOP import get_net
import torchvision.transforms as transforms
from rl_studio.envs.carla.followlane.followlane_env import FollowLaneEnv
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils
from PIL import Image
from scipy.interpolate import interp1d

from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.manual_control import CameraManager
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
    CustomTimer,
)
import pygame
from rl_studio.envs.carla.utils.global_route_planner import (
    GlobalRoutePlanner,
)

from rl_studio.envs.carla.utils.yolop_core.postprocess import morphological_process, connect_lane
from rl_studio.envs.carla.utils.yolop_core.general import non_max_suppression, scale_coords

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

NO_DETECTED = 0


def select_device(logger=None, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            if logger:
                logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        if logger:
            logger.info(f'Using torch {torch.__version__} CPU')

    if logger:
        logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def draw_dash(index, dist, ll_segment):
    ll_segment[index, dist - 1] = 255  # <-- here is the real calculated center
    ll_segment[index, dist - 3] = 255
    ll_segment[index, dist - 2] = 255
    ll_segment[index, dist - 4] = 255
    ll_segment[index, dist - 5] = 255
    ll_segment[index, dist - 6] = 255

def calculate_midpoints(input_array):
    midpoints = []
    for i in range(len(input_array) - 1):
        midpoint = (input_array[i] + input_array[i + 1]) // 2
        midpoints.append(midpoint)
    return midpoints


def add_midpoints(ll_segment, index, dists):
    # Set the value at the specified index and distance to 1
    for dist in dists:
        draw_dash(index, dist, ll_segment)
        draw_dash(index + 2, dist, ll_segment)
        draw_dash(index + 1, dist, ll_segment)
        draw_dash(index - 1, dist, ll_segment)
        draw_dash(index - 2, dist, ll_segment)


def connect_dashed_lines(ll_seg_mask):
    return ll_seg_mask


class FollowLaneStaticWeatherNoTraffic(FollowLaneEnv):
    def __init__(self, **config):

        ###### init class variables
        FollowLaneCarlaConfig.__init__(self, **config)
        self.sync_mode = config["sync"]
        # self.display_manager = None
        # self.vehicle = None
        # self.actor_list = []
        self.timer = CustomTimer()

        self.client = carla.Client(
            config["carla_server"],
            config["carla_client"],
        )
        self.client.set_timeout(10.0)
        print(f"\n maps in carla 0.9.13: {self.client.get_available_maps()}\n")

        self.world = self.client.load_world(config["town"])
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        if self.sync_mode:
            settings.synchronous_mode = True
            self.traffic_manager.set_synchronous_mode(True)
        else:
            self.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(settings)
        current_settings = self.world.get_settings()
        print(f"Current World Settings: {current_settings}")
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

        # INIT YOLOP
        self.yolop_model = get_net()
        self.device = select_device()
        checkpoint = torch.load("/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/utils/weights/End-to-end.pth",
                                map_location=self.device)
        self.yolop_model.load_state_dict(checkpoint['state_dict'])
        self.yolop_model = self.yolop_model.to(self.device)

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
        # self.camera_spectator = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "RGBCamera",
        #    carla.Transform(carla.Location(x=-5, z=2.8), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[0, 0],
        # )
        # self.camera_spectator_segmentated = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "SemanticCamera",
        #    carla.Transform(carla.Location(x=-5, z=2.8), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[1, 0],
        # )
        # self.sergio_camera_spectator = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "SemanticCameraSergio",
        #    carla.Transform(carla.Location(x=-5, z=2.8), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[2, 0],
        # )
        # self.front_camera = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "RGBCamera",
        #    carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[0, 1],
        # )

        # self.front_camera_segmentated = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "SemanticCamera",
        #    carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[1, 1],
        # )

        # self.sergio_front_camera = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "SemanticCameraSergio",
        #    carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[2, 1],
        # )
        # self.front_camera_mas_baja = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "RGBCamera",
        #    carla.Transform(carla.Location(x=2, z=0.5), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[0, 2],
        # )
        # self.front_camera_mas_baja_segmentated = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "SemanticCamera",
        #    carla.Transform(carla.Location(x=2, z=0.5), carla.Rotation(yaw=+00)),
        #    self.car,
        #    {},
        #    display_pos=[1, 2],
        # )

        # self.sergio_front_camera_mas_baja = SensorManager(
        #    self.world,
        #    self.display_manager,
        #    "SemanticCameraSergio",
        #    carla.Transform(carla.Location(x=2, z=0.5), carla.Rotation(yaw=+0)),
        #    self.car,
        #    {},
        #    display_pos=[2, 2],
        # )

        # self.front_camera_1_5_bev = SensorManager(
        #     self.world,
        #     self.display_manager,
        #     "BirdEyeView",
        #     carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
        #     self.car,
        #     {},
        #     display_pos=[1, 1],
        #     client=self.client,
        # )

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

        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        self.set_spectator_location()

        time.sleep(1)
        self.episode_start = time.time()
        self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        AutoCarlaUtils.show_image("image", self.front_camera_1_5.front_camera, 1)

        # mask = self.preprocess_image(
        #    self.front_camera_1_5_red_mask.front_camera_red_mask
        # )
        # AutoCarlaUtils.show_image("mask", mask, 1)

        ## -- states
        # (
        #     states,
        #     distance_center,
        #     distance_to_centr_normalized,
        # ) = self.calculate_states(mask)
        raw_image = self.get_resized_image(self.front_camera_1_5.front_camera)
        ll_segment = self.detect_lines(raw_image)
        # (
        #     distance_center_nop,
        #     _,
        # ) = self.calculate_center(ll_segment)
        # # Iterate over self.x_row and distance_center simultaneously
        # self.show_ll_seg_image(distance_center_nop, ll_segment, "_no_post_process")

        ll_segment_post_process = self.post_process(ll_segment)
        (
            center_lanes,
            distance_to_center_normalized,
        ) = self.calculate_center(ll_segment_post_process)
        right_lane_normalized_distances = [inner_array[0] for inner_array in distance_to_center_normalized]
        right_center_lane = [[inner_array[0]] for inner_array in center_lanes]

        self.show_ll_seg_image(right_center_lane, ll_segment_post_process)

        state_size = len(distance_to_center_normalized)
        time.sleep(1)


        return np.array(right_lane_normalized_distances), state_size

    ####################################################
    ####################################################

    def find_lane_center(self, mask):
        # Find the indices of 1s in the array
        mask_array = np.array(mask)
        indices = np.where(mask_array > 0.8)[0]

        # If there are no 1s or only one set of 1s, return None
        if len(indices) < 2:
            # TODO (Ruben) For the moment we will be dealing with no detection as a fixed number
            return [NO_DETECTED]

        # Find the indices where consecutive 1s change to 0
        diff_indices = np.where(np.diff(indices) > 1)[0]
        # If there is only one set of 1s, return None
        if len(diff_indices) == 0:
            return [NO_DETECTED]

        interested_line_borders = np.array([], dtype=np.int8)
        # print(indices)
        for index in diff_indices:
            interested_line_borders = np.append(interested_line_borders, indices[index])
            interested_line_borders = np.append(interested_line_borders, int(indices[index+1]))
        # print(interested_line_borders)

        # Find the indices of the last 1 in the first set and the first 1 in the second set
        # last_one_first_set = indices[diff_indices[0]]
        # first_one_second_set = indices[diff_indices[0] + 1]
        # Calculate the position of the middle point between the last 1 and the first 1
        # middle_point = (last_one_first_set + first_one_second_set) // 2

        midpoints = calculate_midpoints(interested_line_borders)
        # print(midpoints)
        return midpoints

    def calculate_center(self, mask):
        width = mask.shape[1]
        center_image = width / 2
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        # ## As we drive in the right lane, we get from right to left
        # lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from the right lane to center
        center_lane_indexes = [
            self.find_lane_center(lines[x]) for x, _ in enumerate(lines)
        ]

        center_right_lane_distance = [
            [center_image - x for x in inner_array] for inner_array in center_lane_indexes
        ]

        # Calculate the average position of the right lane lines
        ## normalized distance
        distance_to_center_normalized = [
            np.array(x) / (width - center_image) for x in center_right_lane_distance
        ]
        return center_lane_indexes, distance_to_center_normalized

    def calculate_states(self, mask):
        width = mask.shape[1]
        center_image = width / 2
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
        ## normalized distances
        distance_to_center_normalized = [
            abs(float((center_image - index_right[i]) / center_image))
            for i, _ in enumerate(index_right)
        ]
        # pixels_in_state = mask.shape[1] / self.num_regions
        # states = [int(value / pixels_in_state) for _, value in enumerate(index_right)]
        states = distance_to_center_normalized

        return states, distance_to_center, distance_to_center_normalized

    def preprocess_image(self, red_mask):
        ## first, we cut the upper image
        img_sliced = self.slice_image(red_mask)
        ## -- convert to GRAY
        gray_mask = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        ## --  aplicamos mascara para convertir a BLANCOS Y NEGROS
        _, white_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)

        return white_mask

    def draw_waypoints(self, spawn_points, init, target, lane_id, life_time):
        filtered_waypoints = []
        i = init
        for waypoint in spawn_points[init + 1: target + 2]:
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

    def get_target_waypoint(self, target_waypoint, life_time):
        """
        draw target point
        """
        self.world.debug.draw_string(
            target_waypoint.transform.location,
            "O",
            draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0),
            life_time=life_time,
            persistent_lines=True,
        )

    def setup_car_fix_pose(self, init):
        # while(1):
        #     # Get the current spectator
        #     spectator = self.world.get_spectator()
        #
        #     # Get the transform of the spectator (location and rotation)
        #     spectator_transform = spectator.get_transform()
        #
        #     # Extract the location and rotation
        #     spectator_location = spectator_transform.location
        #     spectator_rotation = spectator_transform.rotation
        #
        #     # Print or use the values as needed
        #     print("Spectator Location:", spectator_location)
        #     print("Spectator Rotation:", spectator_rotation)
        #     time.sleep(10)
        # location = random.choice(self.world.get_map().get_spawn_points())
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = carla.Transform(
            carla.Location(x=-9.732346, y=16.522575, z=10.90110),
            carla.Rotation(pitch=3.958518, yaw=0.607781, roll=0.2),
        )
        # location = carla.Transform(
        #     carla.Location(
        #         x=init.transform.location.x,
        #         y=init.transform.location.y,
        #         z=init.transform.location.z,
        #     ),
        #     carla.Rotation(
        #         pitch=init.transform.rotation.pitch,
        #         yaw=init.transform.rotation.yaw,
        #         roll=init.transform.rotation.roll,
        #     ),
        # )

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

    ################################################################################
    def step(self, action):
        # print(f"=============== STEP ===================")

        ### -------- send action
        params = self.control(action)
        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        # self.set_spectator_location()

        ## -- states
        # mask = self.preprocess_image(
        #    self.front_camera_1_5_red_mask.front_camera_red_mask
        # )
        raw_image = self.get_resized_image(self.front_camera_1_5.front_camera)

        ll_segment = self.detect_lines(raw_image)
        # (
        #     distance_center_nop,
        #     _,
        # ) = self.calculate_center(ll_segment)
        # # Iterate over self.x_row and distance_center simultaneously
        # self.show_ll_seg_image(distance_center_nop, ll_segment, "_no_post_process")

        ll_segment_post_process = self.post_process(ll_segment)
        (
            center_lanes,
            distance_to_center_normalized,
        ) = self.calculate_center(ll_segment_post_process)
        # We get the first of all calculated "center lanes" assuming it will be the right lane
        right_lane_normalized_distances = [inner_array[0] for inner_array in distance_to_center_normalized]
        right_center_lane = [[inner_array[0]] for inner_array in center_lanes]

        self.show_ll_seg_image(right_center_lane, ll_segment_post_process)

        # print(f"states:{states}\n")
        # AutoCarlaUtils.show_image_with_centrals(
        #    "image",
        #    self.front_camera_1_5.front_camera[ll_segment.shape[0] :],
        #    1,
        #    distance_center,
        #    distance_to_center_normalized,
        #    self.x_row,
        # )
        AutoCarlaUtils.show_image("image", self.front_camera_1_5.front_camera, 1)

        ## ------ calculate distance error and states
        # print(f"{self.perfect_distance_normalized =}"
        # right_lane_normalized goes between 1 and -1
        distance_error = [abs(x) for x in right_lane_normalized_distances]
        ## -------- Rewards
        reward, done = self.rewards_easy(distance_error, params)

        return np.array(right_lane_normalized_distances), reward, done, {}

    def control(self, action):

        self.car.apply_control(carla.VehicleControl(throttle=float(action[0]), steer=float(action[1])))
        params = {}

        v = self.car.get_velocity()
        params["velocity"] = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        w = self.car.get_angular_velocity()
        params["steering_angle"] = w

        return params

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

    def rewards_easy(self, distance_error, params):

        done, states_non_line = self.end_if_conditions(distance_error)

        if done:
            return 0, done

        rewards = []
        for _, error in enumerate(distance_error):
            rewards.append(error*10)

        function_reward = sum(rewards) / states_non_line
        function_reward += params["velocity"] * 0.5
        function_reward -= params["steering_angle"].y
        # print("v " + str(params["velocity"]))
        # print("w" + str(params["steering_angle"]))



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

    def slice_image(self, red_mask):
        height = red_mask.shape[0]
        image_middle_line = (height) // 2
        img_sliced = red_mask[image_middle_line:]
        return img_sliced.copy()

    def get_resized_image(self, sensor_data, new_width=640):
        # Assuming sensor_data is the image obtained from the sensor
        # Convert sensor_data to a numpy array or PIL Image if needed
        # For example, if sensor_data is a numpy array:
        # sensor_data = Image.fromarray(sensor_data)
        sensor_data = np.array(sensor_data, copy=True)

        # Get the current width and height
        height = sensor_data.shape[0]
        width = sensor_data.shape[1]

        # Calculate the new height to maintain the aspect ratio
        new_height = int((new_width / width) * height)

        resized_img = Image.fromarray(sensor_data).resize((new_width, new_height))

        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        resized_img_np = np.array(resized_img)

        return resized_img_np

    def detect_lines(self, raw_image):
        with torch.no_grad():
            return self.detect(raw_image)

    def detect(self, raw_image):
        # Get names and colors
        names = self.yolop_model.module.names if hasattr(self.yolop_model, 'module') else self.yolop_model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        img = transform(raw_image).to(self.device)
        img = img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        det_out, da_seg_out, ll_seg_out = self.yolop_model(img)
        # det_pred = non_max_suppression(det_out, classes=None, agnostic=False)
        # det = det_pred[0]

        # _, _, height, width = img.shape
        # h, w, _ = img_det.shape
        # pad_w, pad_h = shapes[1][1]
        # pad_w = int(pad_w)
        # pad_h = int(pad_h)
        # ratio = shapes[1][0][1]
        #
        # da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        # da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        # _, da_seg_mask = torch.max(da_seg_mask, 1)
        # da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
        #
        # ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=int(1), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        return ll_seg_mask

    def show_ll_seg_image(self, dists, ll_segment, suffix=""):
        ll_segment_int8 = (ll_segment * 255).astype(np.uint8)
        ll_segment_all = [np.copy(ll_segment_int8),np.copy(ll_segment_int8),np.copy(ll_segment_int8)]
        # Iterate over self.x_row and distance_center simultaneously
        for index, dist in zip(self.x_row, dists):
            # Set the value at the specified index and distance to 1
            add_midpoints(ll_segment_all[0], index, dist)

        # draw a line for the selected perception points
        for index in self.x_row:
            for i in range(630):
                ll_segment_all[0][index][i] = 255

        ll_segment_stacked = np.stack(ll_segment_all, axis=-1)
        # We now show the segmentation and center lane postprocessing
        cv2.imshow('ll_seg' + suffix, ll_segment_stacked)
        # cv2.imshow('image', det)
        cv2.waitKey(1)  # 1 millisecond

    def post_process(self, ll_segment):
        # # Lane line post-processing
        ll_seg_mask = morphological_process(ll_segment, kernel_size=5, func_type=cv2.MORPH_OPEN)
        ll_seg_mask = connect_dashed_lines(ll_seg_mask)

        # ll_seg_mask = connect_lane(ll_seg_mask)
        #
        # img_det =  show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
        return ll_seg_mask

    def set_spectator_location(self):
        spectator = self.world.get_spectator()

        world_snapshot = self.world.get_snapshot()
        actor_snapshot = world_snapshot.find(self.car.id)
        # Set spectator at given transform (vehicle transform)
        # spectator.set_transform(actor_snapshot.get_transform())

        # Get the car's current transform
        car_transform = actor_snapshot.get_transform()

        # Calculate the pitch and yaw angles to look down at the road
        pitch = -90  # Look down
        yaw = car_transform.rotation.yaw  # Maintain the same yaw angle as the car

        # Create a rotation quaternion from Euler angles
        rotation = carla.Rotation(pitch, yaw, 0)
        # Set the spectator's transform with adjusted z-coordinate
        new_spectator_transform = carla.Transform(car_transform.location + carla.Location(z=100),
                                                  rotation)

        spectator.set_transform(new_spectator_transform)

    def end_if_conditions(self, right_lane_normalized_distances):
        counter_states = Counter(right_lane_normalized_distances)
        states_non_line = counter_states.get(1)
        ## ----- hacemos la salida
        done = False
        if states_non_line is not None and (
                states_non_line > len(right_lane_normalized_distances) // 1.5
        ):  # salimos porque no detecta linea a la derecha
            done = True
        if len(self.collision_hist) > 0:  # te has chocado, baby
            done = True

        return done, states_non_line

