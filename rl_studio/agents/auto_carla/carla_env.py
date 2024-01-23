import math
import random
import time

import carla
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from rl_studio.agents.auto_carla.actors_sensors import LaneDetector
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig


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


# ==============================================================================
# -- class CarlaEnv -------------------------------------------------------------
# ==============================================================================


class CarlaEnv(gym.Env):
    # def __init__(self, car, sensor_camera_rgb, sensor_camera_lanedetector, config):
    def __init__(self, car, sensor_camera_rgb, config):
        super(CarlaEnv, self).__init__()
        ## --------------- init env
        # FollowLaneEnv.__init__(self, **config)
        ## --------------- init class variables
        FollowLaneCarlaConfig.__init__(self, **config)

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

        # print(f"{self.alternate_pose =}\n")
        # print(f"{self.start_alternate_pose =}\n")

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

        #########################################################################

        # self.params = config
        # print(f"{self.params =}\n")

        self.car = car
        self.sensor_camera_rgb = sensor_camera_rgb
        # self.sensor_camera_lanedetector = sensor_camera_lanedetector
        self.params = {}
        self.is_finish = None
        self.dist_to_finish = None
        self.collision_hist = []

        self.right_line_in_pixels = None
        self.lane_centers_in_pixels = None
        self.ground_truth_pixel_values = None
        self.dist_normalized = None
        self.states = None
        self.state_right_lines = None
        self.drawing_lines_states = []
        self.drawing_numbers_states = []

        self.lane_changing_hist = []

        ######################################################################

        ## LaneDetector Camera ---------------
        self.sensor_camera_lanedetector = LaneDetector(
            "models/fastai_torch_lane_detector_model.pth"
        )

        # torch.cuda.empty_cache()
        # self.model = torch.load("models/fastai_torch_lane_detector_model.pth")
        # self.model.eval()

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

        ############################################################################
        ########### --- calculating STATES

        ##### EATA PARTE NOS SIRVE PARA ENCONTRAR EL CENTRO SOLO CON LA LINEA DERECHA
        while self.sensor_camera_rgb.front_rgb_camera is None:
            print(f"RESET() ----> {self.sensor_camera_rgb.front_rgb_camera = }")
            time.sleep(0.2)

        # TODO: spawning car

        self.states = [self.num_regions // 2 for i, _ in enumerate(self.x_row)]
        print(f"\n\t{self.states =}")
        # states_size = len(self.states)

        # print(f"{states =} and {states_size =}")
        return self.states  # , states_size

    #####################################################################################
    #
    #                                       STEP
    #
    #####################################################################################
    def step(self, action):
        """
        state: sp
        actions: discretes
                LaneDetector
        """

        self.control_discrete_actions(action)
        # print(f"\n\tSTEP() POR AQUI")

        """
            WORKING WITH ONLY RIGHT LATERAL LINE
        ########### --- calculating STATES
        mask = self.preprocess_image(self.sensor_camera_red_mask.front_red_mask_camera)

        ########### --- Calculating center ONLY with right line
        self.right_line_in_pixels = self.calculate_right_line(mask, self.x_row)

        image_center = mask.shape[1] // 2
        # dist = [
        #    image_center - right_line_in_pixels[i]
        #    for i, _ in enumerate(right_line_in_pixels)
        # ]
        self.dist_normalized = [
            float((image_center - self.right_line_in_pixels[i]) / image_center)
            for i, _ in enumerate(self.right_line_in_pixels)
        ]

        """

        ###################################### ORIGINAL LANEDETECTOR

        back, left, right = self.sensor_camera_lanedetector.get_prediction(
            self.sensor_camera_rgb.front_rgb_camera
        )[0]
        image_rgb_lanedetector = self.sensor_camera_lanedetector.lane_detection_overlay(
            self.sensor_camera_rgb.front_rgb_camera, left, right
        )

        AutoCarlaUtils.show_image(
            "LaneDetector RGB",
            # image_rgb_lanedetector[(image_rgb_lanedetector.shape[0] // 2) :],
            image_rgb_lanedetector[(image_rgb_lanedetector.shape[0] // 2) :],
            600,
            800,
        )

        ####################################### LANEDETECTOR + REGRESSION

        image_copy = np.copy(self.sensor_camera_rgb.front_rgb_camera)
        (
            image_rgb_lanedetector_regression,
            left_mask,
            right_mask,
        ) = self.sensor_camera_lanedetector.detect(image_copy)

        AutoCarlaUtils.show_image(
            "LaneDetector RGB Regression",
            # image_rgb_lanedetector[(image_rgb_lanedetector.shape[0] // 2) :],
            image_rgb_lanedetector_regression[
                (image_rgb_lanedetector_regression.shape[0] // 2) :
            ],
            1400,
            800,
        )

        ####################################### ONLY RIGHT LINE

        #######################################

        mask = self.preprocess_image_lane_detector(image_rgb_lanedetector_regression)

        AutoCarlaUtils.show_image(
            "mask",
            mask,
            300,
            10,
        )

        (
            lane_centers_in_pixels,
            errors,
            errors_normalized,
            index_right,
            index_left,
        ) = self.calculate_lane_centers_with_lane_detector(mask)

        print(
            f"\n\t{lane_centers_in_pixels = }\n\t{errors =}\n{errors_normalized =}\n\t{index_right = }\n\t{index_left =}"
        )

        ## STATES
        # pixels_in_state = mask.shape[1] / self.num_regions
        # self.states = [
        #    int(value / pixels_in_state)
        #    for _, value in enumerate(lane_centers_in_pixels)
        # ]

        # pixels_in_state = mask.shape[1] // self.num_regions
        # self.state_right_lines = [
        #    i for i in range(1, mask.shape[1]) if i % pixels_in_state == 0
        # ]
        # self.states = [
        #    int(value / pixels_in_state)
        #    for _, value in enumerate(self.right_line_in_pixels)
        # ]

        """
        image_center = mask.shape[1] // 2
        dist = [
            image_center - self.lane_centers_in_pixels[i]
            for i, _ in enumerate(self.lane_centers_in_pixels)
        ]
        self.dist_normalized = [
            float((image_center - self.lane_centers_in_pixels[i]) / image_center)
            for i, _ in enumerate(self.lane_centers_in_pixels)
        ]
        """

        ############################################
        #
        #   DRWAWING LINES
        #############################################

        (
            self.states,
            drawing_lines_states,
            drawing_numbers_states,
        ) = self.calculate_states(
            mask.shape[1],
            lane_centers_in_pixels,
            self.num_regions,
            size_lateral_states=140,
        )
        print(
            f"\n\t{self.states = }\n\t{drawing_lines_states =}\n\t{drawing_numbers_states =}"
        )
        AutoCarlaUtils.show_image_lines_centers_borders(
            "Front Camera",
            self.sensor_camera_rgb.front_rgb_camera[
                (self.sensor_camera_rgb.front_rgb_camera.shape[0] // 2) :
            ],
            self.x_row,
            800,
            10,
            index_right,
            index_left,
            lane_centers_in_pixels,
            drawing_lines_states,
            drawing_numbers_states,
        )

        ## -------- Ending Step()...
        ###################################
        #
        #       REWARDS
        ###################################

        done = False

        """
        esto que viene es para ONLY RIGHT LINE
        
        
        ground_truth_normal_values = [
            correct_normalized_distance[value] for i, value in enumerate(self.x_row)
        ]
        # reward, done = self.autocarlarewards.rewards_right_line(
        #    dist_normalized, ground_truth_normal_values, self.params
        # )
        reward, done = self.autocarlarewards.rewards_sigmoid_only_right_line(
            self.dist_normalized, ground_truth_normal_values
        )
        """
        reward, done = self.autocarlarewards.rewards_followlane_two_lines(
            errors_normalized, self.center_image, self.params
        )

        #################################
        #
        #       any OVER causes
        #################################

        ## -------- ... or Finish by...
        if len(self.collision_hist) > 0:  # crashed you, baby
            done = True
            # reward = -100
            print(f"crash")

        self.is_finish, self.dist_to_finish = AutoCarlaUtils.finish_fix_number_target(
            self.params["location"],
            self.finish_alternate_pose,
            self.finish_pose_number,
            self.max_target_waypoint_distance,
        )
        if self.is_finish:
            print(f"Finish!!!!")
            done = True

        # self.ground_truth_pixel_values = [
        #    correct_pixel_distance[value] for i, value in enumerate(self.x_row)
        # ]

        """
        AutoCarlaUtils.show_image(
            "mask",
            mask,
            50,
            500,
        )

        AutoCarlaUtils.show_image(
            "LaneDetector RGB",
            image_rgb_lanedetector[(image_rgb_lanedetector.shape[0] // 2) :],
            600,
            500,
        )

        AutoCarlaUtils.show_image(
            "front RGB",
            self.sensor_camera_rgb.front_rgb_camera[
                (self.sensor_camera_rgb.front_rgb_camera.shape[0] // 2) :
            ],
            1250,
            500,
        )
        """

        return self.states, reward, done, {}

    ##################################################
    #
    #   Control
    ###################################################

    def control_discrete_actions(self, action):
        """
        working with LaneDetector
        """
        t = self.car.car.get_transform()
        v = self.car.car.get_velocity()  # returns in m/sec
        c = self.car.car.get_control()
        w = self.car.car.get_angular_velocity()  # returns in deg/sec
        a = self.car.car.get_acceleration()
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
        throttle = self.actions[action][0]
        steer = self.actions[action][1]
        # print(f"in STEP() {throttle = } and {steer = }")

        target_speed = 30
        brake = 0

        # Limiting Velocity up to target_speed = 30 (or any other)
        if curr_speed > target_speed:
            throttle = 0.45

        self.car.car.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

    #####################################################################################
    # ---   methods
    #####################################################################################

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

        index_right = [
            mask.shape[1] - inv_index_right[x] if inv_index_right[x] != 0 else 0
            for x, _ in enumerate(inv_index_right)
        ]

        return index_right

    def preprocess_image_lane_detector(self, image):
        """
        image from lane detector with regression
        """
        ## first, we cut the upper image
        height = image.shape[0]
        image_middle_line = (height) // 2
        img_sliced = image[image_middle_line:]

        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(img_sliced, kernel, iterations=3)
        hsv = cv2.cvtColor(img_erosion, cv2.COLOR_RGB2HSV)
        gray_hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
        _, gray_mask = cv2.threshold(gray_hsv, 100, 255, cv2.THRESH_BINARY)

        return gray_mask

    def calculate_lane_centers_with_lane_detector(self, gray_mask):
        """
        using Lane Detector model for calculating the center
        """
        lines = [gray_mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]

        # lines_conoffset = [lines[i][offset:] for i,_ in enumerate(lines)]
        # print(f"{lines=}")
        # print(f"{lines_conoffset=}")
        index_left = [np.argmax(lines[x]) for x, _ in enumerate(lines)]
        # print(f"{index_left = }")

        # calculamos la linea derecha desde la columna 640, desde la derecha
        lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        # print(f"{lines_inversed = }")
        inv_index_right = [
            np.argmax(lines_inversed[x]) for x, _ in enumerate(lines_inversed)
        ]
        # print(f"{inv_index_right = }")
        index_right = [
            gray_mask.shape[1] - inv_index_right[x] if inv_index_right[x] != 0 else 0
            for x, _ in enumerate(inv_index_right)
        ]
        # print(f"{index_right = }")

        centers = [
            ((right - left) // 2) + left for right, left in zip(index_right, index_left)
        ]
        errors = [
            abs(centers[i] - gray_mask.shape[1] // 2) for i, _ in enumerate(centers)
        ]
        errors_normalized = [
            float(((gray_mask.shape[1] // 2) - centers[i]) / (gray_mask.shape[1] // 2))
            for i, _ in enumerate(errors)
        ]
        return centers, errors, errors_normalized, index_right, index_left

    def calculate_states(
        self, width, lane_centers_in_pixels, num_regions, size_lateral_states
    ):
        # size_lateral_states = 140
        size_center_states = width - (size_lateral_states * 2)
        pixel_center_states = int(size_center_states / (num_regions - 2))

        states = [
            int(((value - size_lateral_states) / pixel_center_states) + 1)
            if (width - size_lateral_states) > value > size_lateral_states
            else num_regions - 1
            if value >= (width - size_lateral_states)
            else num_regions
            for _, value in enumerate(lane_centers_in_pixels)
        ]

        # drawing lines and numbers states in image
        drawing_lines_states = [
            size_lateral_states + (i * pixel_center_states)
            for i in range(1, num_regions - 1)
        ]
        # self.drawing_lines_states.append(size_lateral_states)
        drawing_lines_states.insert(0, size_lateral_states)

        drawing_numbers_states = [
            i if i > 0 else num_regions for i in range(0, num_regions)
        ]

        return states, drawing_lines_states, drawing_numbers_states

    """
    def OLD_calculate_lane_centers_with_lane_detector(self, mask):
        
        #using Lane Detector model for calculating the center
        
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        index_left = [np.argmax(lines[x]) for x, _ in enumerate(lines)]
        left_offset, right_offset = 20, 10
        index_left_plus_offset = [
            index_left[x] + left_offset if index_left[x] != 0 else 0
            for x, _ in enumerate(index_left)
        ]
        second_lines_from_left_right = [
            lines[x][index_left_plus_offset[x] :] for x, _ in enumerate(lines)
        ]
        ## ---------------- calculating index in second line
        try:
            index_right_no_offset = [
                np.argmax(second_lines_from_left_right[x])
                for x, _ in enumerate(second_lines_from_left_right)
            ]
        except:
            secod_lines_from_left_right = [value for _, value in enumerate(self.x_row)]
            index_right_no_offset = [
                np.argmax(second_lines_from_left_right[x])
                for x, _ in enumerate(second_lines_from_left_right)
            ]

        index_right_plus_offsets = [
            index_right_no_offset[x] + left_offset + right_offset
            for x, _ in enumerate(index_right_no_offset)
        ]

        index_right = [
            right + left for right, left in zip(index_right_plus_offsets, index_left)
        ]

        centers = [
            ((right - left) // 2) + left for right, left in zip(index_right, index_left)
        ]

        return centers, index_left, index_right

        """

    """
    ############### VAMOS A PROBAR EL ORIGINAL
    @staticmethod
    def get_prediction(model, img_array):
        with torch.no_grad():
            image_tensor = img_array.transpose(2, 0, 1).astype("float32") / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            model_output = torch.softmax(model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output

    @staticmethod
    def lane_detection_overlay(image, left_mask, right_mask):
        res = np.copy(image)
        # We show only points with probability higher than 0.5
        res[left_mask > 0.07, :] = [255, 0, 0]
        res[right_mask > 0.07, :] = [0, 0, 255]

        cv2.erode(left_mask, (7, 7), 4)
        cv2.dilate(left_mask, (7, 7), 4)

        return res

    # @staticmethod
    def detect(self, model, img_array: np.array) -> tuple:
        with torch.no_grad():
            image_tensor = img_array.transpose(2, 0, 1).astype("float32") / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            back, left, right = (
                torch.softmax(model.forward(x_tensor), dim=1).cpu().numpy()[0]
            )

        res, left_mask, right_mask = self.lane_detection_overlay(img_array, left, right)

        return res, left_mask, right_mask

    """

    '''
    def lane_detection_overlay(
        self, image: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray
    ) -> tuple:
        """
        @type left_mask: object
        """
        res = np.copy(image)

        cv2.erode(left_mask, (7, 7), 4)
        cv2.dilate(left_mask, (7, 7), 4)

        cv2.erode(right_mask, (7, 7), 4)
        cv2.dilate(right_mask, (7, 7), 4)

        left_mask = self.image_polyfit(left_mask)
        right_mask = self.image_polyfit(right_mask)

        # We show only points with probability higher than 0.07
        res[left_mask > 0.1, :] = [255, 0, 0]
        res[right_mask > 0.1, :] = [0, 0, 255]

        return res, left_mask, right_mask

    '''

    """
    def image_polyfit(self, image: np.ndarray) -> np.ndarray:
        img = np.copy(image)
        img[image > 0.1] = 255

        indices = np.where(img == 255)

        if len(indices[0]) == 0:
            return img
        grade = 1
        coefficients = np.polyfit(indices[0], indices[1], grade)

        x = np.linspace(0, img.shape[1], num=2500)
        y = np.polyval(coefficients, x)
        points = np.column_stack((x, y)).astype(int)

        valid_points = []

        for point in points:
            # if (0 < point[1] < 1023) and (0 < point[0] < 509):
            if (0 < point[1] < image.shape[1]) and (0 < point[0] < image.shape[0]):
                valid_points.append(point)

        valid_points = np.array(valid_points)
        polyfitted = np.zeros_like(img)
        polyfitted[tuple(valid_points.T)] = 255

        return polyfitted

    """
