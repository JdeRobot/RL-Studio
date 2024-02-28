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
    for i in range(0, len(input_array) - 1, 2):
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
    # TODO
    return ll_seg_mask

def discard_not_confident_centers(center_lane_indexes):
    # Count the occurrences of each list size leaving out of the equation the non-detected
    size_counter = Counter(len(inner_list) for inner_list in center_lane_indexes if NO_DETECTED not in inner_list)
    # Check if size_counter is empty, which mean no centers found
    if not size_counter:
        return center_lane_indexes
    # Find the most frequent size
    # most_frequent_size = max(size_counter, key=size_counter.get)

    # Iterate over inner lists and set elements to 1 if the size doesn't match majority
    result = []
    for inner_list in center_lane_indexes:
        # if len(inner_list) != most_frequent_size:
        if len(inner_list) != 2: # If we don't see the 2 lanes, we discard the row
            inner_list = [NO_DETECTED] * len(inner_list)  # Set all elements to 1
        result.append(inner_list)

    return result


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
        spectator = self.world.get_spectator()
        spectator_location = carla.Transform(
            location.location + carla.Location(z=100),
            carla.Rotation(-90, location.rotation.yaw, 0))
        spectator.set_transform(spectator_location)

        time.sleep(1)

    def reset(self):

        self.collision_hist = []
        self.actor_list = []

        self.set_init_pose()
        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        time.sleep(1)
        self.episode_start = time.time()
        self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        AutoCarlaUtils.show_image("image", self.front_camera_1_5.front_camera, 1)


        raw_image = self.get_resized_image(self.front_camera_1_5.front_camera)

        ll_segment = self.detect_lines(raw_image)
        ll_segment_post_process = self.post_process(ll_segment)
        (
            center_lanes,
            distance_to_center_normalized,
        ) = self.calculate_center(ll_segment_post_process)
        right_lane_normalized_distances = [inner_array[-1] for inner_array in distance_to_center_normalized]
        right_center_lane = [[inner_array[-1]] for inner_array in center_lanes]

        self.show_ll_seg_image(right_center_lane, ll_segment_post_process)

        state_size = len(distance_to_center_normalized)
        # right_lane_normalized_distances = [1,1,1,1,1,1,1,1,1,1]
        # state_size = 12
        time.sleep(1)
        right_lane_normalized_distances.append(0)
        right_lane_normalized_distances.append(0)

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

        # this part consists of checking the number of lines detected in all rows
        # then discarding the rows (set to 1) in which more or less centers are detected
        center_lane_indexes = discard_not_confident_centers(center_lane_indexes)

        center_lane_distances = [
            [center_image - x for x in inner_array] for inner_array in center_lane_indexes
        ]

        # Calculate the average position of the right lane lines
        ## normalized distance
        distance_to_center_normalized = [
            np.array(x) / (width - center_image) for x in center_lane_distances
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
        # i = init
        # for waypoint in spawn_points[init + 1: target + 2]:
        i = 1
        for waypoint in spawn_points:
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
        right_lane_normalized_distances = [inner_array[-1] for inner_array in distance_to_center_normalized]
        right_center_lane = [[inner_array[-1]] for inner_array in center_lanes]

        self.show_ll_seg_image(right_center_lane, ll_segment_post_process)
        # self.show_ll_seg_image(center_lanes, ll_segment_post_process, name="ll_seg_all")

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
        right_lane_normalized_distances.append(params["velocity"]/5)
        right_lane_normalized_distances.append(params["steering_angle"])

        return np.array(right_lane_normalized_distances), reward, done, params

    def control(self, action):

        self.car.apply_control(carla.VehicleControl(throttle=float(action[0]), steer=float(action[1])))
        params = {}

        v = self.car.get_velocity()
        params["velocity"] = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        w = self.car.get_angular_velocity()
        params["angular_velocity"] = w

        w_angle = self.car.get_control().steer
        params["steering_angle"] = w_angle

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

        done, states_non_line = self.end_if_conditions(distance_error, threshold=0.3)

        if done:
            params["d_reward"] = 0
            params["v_reward"] = 0
            params["v_eff_reward"] = 0
            params["reward"] = 0
            return -self.punish_ineffective_vel, done

        d_rewards = []
        for _, error in enumerate(distance_error):
            d_rewards.append(pow(1 - error, 5))

        d_reward = sum(d_rewards) / ( len(distance_error) - states_non_line )
        params["d_reward"] = d_reward
        # reward Max = 1 here
        punish = 0
        if params["velocity"] < 1.5:
            punish += self.punish_ineffective_vel
        punish += self.punish_zig_zag_value * params["steering_angle"]

        v_reward = params["velocity"] / 5
        v_eff_reward = v_reward * pow(d_reward, 2)
        params["v_reward"] = v_reward
        params["v_eff_reward"] = v_eff_reward

        beta = self.beta
        # TODO Ver que valores toma la velocity para compensarlo mejor
        function_reward = beta * d_reward + (1-beta) * v_eff_reward
        function_reward -= punish
        params["reward"] = function_reward

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

        # Run inference
        img = transform(raw_image).to(self.device)
        img = img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        det_out, da_seg_out, ll_seg_out = self.yolop_model(img)

        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=int(1), mode='bicubic')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        return ll_seg_mask

    def show_ll_seg_image(self,dists, ll_segment, suffix="",  name='ll_seg'):
        ll_segment_int8 = (ll_segment * 255).astype(np.uint8)
        ll_segment_all = [np.copy(ll_segment_int8),np.copy(ll_segment_int8),np.copy(ll_segment_int8)]

        # draw the midpoint used as right center lane
        for index, dist in zip(self.x_row, dists):
            # Set the value at the specified index and distance to 1
            add_midpoints(ll_segment_all[0], index, dist)

        # draw a line for the selected perception points
        for index in self.x_row:
            for i in range(630):
                ll_segment_all[0][index][i] = 255

        ll_segment_stacked = np.stack(ll_segment_all, axis=-1)
        # We now show the segmentation and center lane postprocessing
        cv2.imshow(name + suffix, ll_segment_stacked)
        cv2.waitKey(1)  # 1 millisecond

    def post_process(self, ll_segment):
        ''''
        Lane line post-processing
        '''
        ll_segment = morphological_process(ll_segment, kernel_size=5, func_type=cv2.MORPH_OPEN)
        ll_segment = morphological_process(ll_segment, kernel_size=20, func_type=cv2.MORPH_CLOSE)
        return ll_segment

    def post_process_hough(self, ll_segment):
        ll_segment_int8 = (ll_segment * 255).astype(np.uint8)
        # Detect lines using HoughLines
        lines = cv2.HoughLinesP(
            ll_segment_int8,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 60,  # Angle resolution in radians
            threshold=80,  # Min number of votes for valid line
            minLineLength=10,  # Min allowed length of line
            maxLineGap=60  # Max allowed gap between line for joining them
        )

        # Filter and draw the two most prominent lines
        if lines is None:
            ll_segment = morphological_process(ll_segment, kernel_size=5, func_type=cv2.MORPH_OPEN)
            ll_segment = morphological_process(ll_segment, kernel_size=20, func_type=cv2.MORPH_CLOSE)
            return ll_segment

        # Sort lines by their length
        # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

        # Create a blank image to draw lines
        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Iterate over points
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Postprocess the detected lines
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
        # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
        # eroded_image = cv2.erode(line_mask, kernel, iterations=1)
        ll_segment_int8 = (line_mask // 255).astype(np.uint8)
        # TODO We could still pass the houghlines to the final image to check if we can extends the detected
        # lines with some arithmetics
        return ll_segment_int8

    def extend_lines(self, lines, image_height):
        extended_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope and intercept
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                # Calculate new endpoints to extend the line
                x1_extended = int(x1 - 2 * (x2 - x1))  # Extend 2 times the original length
                y1_extended = int(slope * x1_extended + intercept)
                x2_extended = int(x2 + 2 * (x2 - x1))  # Extend 2 times the original length
                y2_extended = int(slope * x2_extended + intercept)
                # Ensure the extended points are within the image bounds
                x1_extended = max(0, min(x1_extended, image_height - 1))
                y1_extended = max(0, min(y1_extended, image_height - 1))
                x2_extended = max(0, min(x2_extended, image_height - 1))
                y2_extended = max(0, min(y2_extended, image_height - 1))
                # Append the extended line to the list
                extended_lines.append([(x1_extended, y1_extended, x2_extended, y2_extended)])
        return extended_lines

    def end_if_conditions(self, distances_error, threshold=0.6, min_conf_states=2):
        done = False

        states_above_threshold = sum(1 for state_value in distances_error if state_value > threshold)

        if states_above_threshold is None:
            states_above_threshold = 0

        if (states_above_threshold > len(distances_error) - min_conf_states):  # salimos porque no detecta linea a la derecha
            done = True
        if len(self.collision_hist) > 0:  # te has chocado, baby
            done = True

        return done, states_above_threshold

    def set_init_pose(self):
        ## ---  Car
        waypoints_town = self.world.get_map().generate_waypoints(5.0)
        init_waypoint = waypoints_town[self.waypoints_init + 1]

        if self.alternate_pose:
            self.setup_car_random_pose()
        elif self.waypoints_init is not None:
            # self.draw_waypoints(
            #     waypoints_town,
            #     self.waypoints_init,
            #     self.waypoints_target,
            #     self.waypoints_lane_id,
            #     2000,
            # )
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


