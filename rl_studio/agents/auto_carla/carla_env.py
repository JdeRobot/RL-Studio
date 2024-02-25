import math
import random
import time

import carla
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from rl_studio.agents.auto_carla.actors_sensors import LaneDetector
from rl_studio.agents.auto_carla.utils import AutoCarlaUtils
from rl_studio.agents.auto_carla.settings import FollowLaneCarlaConfig


correct_normalized_distance = {  # based in an image of 640 pixels width
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
correct_pixel_distance = {  # based in an image of 640 pixels width
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
    def __init__(self, car, sensor_camera_rgb, sensor_camera_red_mask, config):
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

        ######## observations Gym based
        # image
        if self.state_space == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )
        else:  # Discrete observations vector = [0.0, 3.6, 8.9]
            # TODO: change x_row for other list
            self.observation_space = spaces.Discrete(len(self.x_row))  # temporary

        # print(f"\n\t In case of implementing Stable-Baselines-gym-based:")
        # print(
        #    f"\n\tIn CarlaEnv --> {self.state_space =}, {self.actions_space =}, {self.action_space =}, {self.observation_space =}"
        # )

        #########################################################################

        # self.params = config
        # print(f"{self.params =}\n")

        self.car = car
        self.sensor_camera_rgb = sensor_camera_rgb
        self.sensor_camera_red_mask = sensor_camera_red_mask
        # self.sensor_camera_lanedetector = sensor_camera_lanedetector
        self.params = {}
        self.is_finish = None
        self.dist_to_finish = None
        self.collision_hist = []

        self.right_line_in_pixels = None
        self.lane_centers_in_pixels = None
        self.ground_truth_pixel_values = None
        self.dist_normalized = None
        self.state = None
        self.state_right_lines = None
        self.drawing_lines_states = []
        self.drawing_numbers_states = []

        self.lane_changing_hist = []
        self.target_veloc = config["target_vel"]
        self.angle = None
        self.centers_normal = []

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
        In Reset() and step():
        IMPLEMENTING STATES FOR SIMPLIFIED PERCEPTION
        STATES = V + W + ANGLE + CENTERS + LINE BORDERS (+ POSE_X + POSE_Y OPTIONALS)
        NUMBER OF PARAMETERS = X_ROW * 3 + 3 WHICH IS THE INPUT_SIZE IN NN
        FOR RESET() :
              V = 0
              W = 0
              ANGLE = 0
              CENTERS = STATES IN PIXELS (DEPENDS ON X_ROW)
              LINE BORDERS =
              POSE_X, POSE_Y = SPAWNING CAR POINTS IN THE MAP (IT IS A POSSIBILITY, NOT IMPLEMENTED RIGHT NOW)

        EXAMPLE -> states = [V=0, W=0, ANGLE=0, CENTERS=300, 320, 350, LINE_BORDERS = 200, 180, 150, 350, 400, 500]
                                  12 VALUES WHEN X_ROW = 3

        """

        while self.sensor_camera_rgb.front_rgb_camera is None:
            print(
                f"\n\treset() ----> {self.sensor_camera_rgb.front_rgb_camera = } ---> waiting for image"
            )
            time.sleep(0.2)

        # TODO: spawning car

        #### LANEDETECTOR + REGRESSION
        image_copy = np.copy(self.sensor_camera_rgb.front_rgb_camera)
        (
            image_rgb_lanedetector_regression,
            _,
            _,
        ) = self.sensor_camera_lanedetector.detect(image_copy)

        mask = self.preprocess_image_lane_detector(image_rgb_lanedetector_regression)

        (
            lane_centers_in_pixels,
            errors,
            errors_normalized,
            index_right,
            index_left,
        ) = self.calculate_lane_centers_with_lane_detector(mask)

        ### CALCULATING STATES AS [lane_centers_in_pixels, index_right, index_left, V, W, ANGLE]
        self.state = self.DQN_states_simplified_perception_normalized(
            0,
            self.target_veloc,
            0,
            0,
            lane_centers_in_pixels,
            index_right,
            index_left,
            mask.shape[1],
        )

        return self.state  # , states_size

    #####################################################################################
    #
    #                                       STEP
    #
    #####################################################################################

    def step(self, action):

        lane_detector_pytorch = True
        if lane_detector_pytorch:
            return self.step_lane_detector_pytorch(action)

        else:
            return self.step_only_right_line(action)

    def step_lane_detector_pytorch(self, action):
        """
        state: sp
        actions: discretes
                LaneDetector
        """

        self.control_discrete_actions(action)

        ####################################### LANEDETECTOR + REGRESSION
        image_copy = np.copy(self.sensor_camera_rgb.front_rgb_camera)
        (
            image_rgb_lanedetector_regression,
            left_mask,
            right_mask,
        ) = self.sensor_camera_lanedetector.detect(image_copy)

        mask = self.preprocess_image_lane_detector(image_rgb_lanedetector_regression)
        AutoCarlaUtils.show_image(
            "mask",
            # mask,
            cv2.resize(mask, (400, 200), cv2.INTER_AREA),
            10,
            10,
        )
        AutoCarlaUtils.show_image(
            "RGB",
            # image_copy,
            cv2.resize(image_copy, (400, 200), cv2.INTER_AREA),
            1500,
            10,
        )
        AutoCarlaUtils.show_image(
            "LaneDetector",
            # image_rgb_lanedetector_regression[
            #    (image_rgb_lanedetector_regression.shape[0] // 2) :
            # ],
            cv2.resize(
                image_rgb_lanedetector_regression[
                    (image_rgb_lanedetector_regression.shape[0] // 2) :
                ],
                (400, 200),
                cv2.INTER_AREA,
            ),
            1100,
            10,
        )
        (
            lane_centers_in_pixels,
            errors,
            self.centers_normal,
            index_right,
            index_left,
        ) = self.calculate_lane_centers_with_lane_detector(mask)

        # print(
        #    f"\n\tin step()...\n\t{lane_centers_in_pixels = }\n\t{errors =}\n\t{errors_normalized =}\n\t{index_right = }\n\t{index_left =}"
        # )

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
        # print(
        # f"\n\t{self.states = }\n\t{drawing_lines_states =}\n\t{drawing_numbers_states =}"
        #    f"\n\t{self.states = }"
        # )
        AutoCarlaUtils.show_image_lines_centers_borders(
            "Front Camera",
            self.sensor_camera_rgb.front_rgb_camera[
                (self.sensor_camera_rgb.front_rgb_camera.shape[0] // 2) :
            ],
            self.x_row,
            450,
            10,
            index_right,
            index_left,
            lane_centers_in_pixels,
            drawing_lines_states,
            drawing_numbers_states,
        )

        ################# Heading
        angle_height_image = self.sensor_camera_rgb.front_rgb_camera.shape[0] // 2
        angle_weight_image = self.sensor_camera_rgb.front_rgb_camera.shape[1] // 2
        central_line_points = [
            (angle_weight_image, 0),
            [angle_weight_image, angle_height_image],
        ]
        # centers_lines_points = [(320, 20), (340, 160), (360, 300), (400, 480)]
        centers_lines_points = list(zip(lane_centers_in_pixels, self.x_row))
        self.angle = self.heading_car(central_line_points, centers_lines_points)
        # print(f"\n\t{angle =}")
        # input("press ...")

        ################################################### RESET by:
        ## 1. No lateral lines detected
        ## 2. REWARD: far from center, vel > target, heading > 30
        ## 3. reach FINISH line
        ## 4. Collision
        ##

        ## -------- Step() over...
        done = False
        # print(
        #    f"\n\t{errors_normalized = }\n\t{self.params['current_speed']=}"
        #    f"\n\t{self.params['current_steering_angle']=},\n\t{self.params['target_veloc']=},"
        #    f"\n\t{angle=}"
        # )

        # reward, done, centers_rewards_list = (
        (
            center_reward,
            done_center,
            centers_rewards_list,
            velocity_reward,
            done_velocity,
            heading_reward,
            done_heading,
            done,
            reward,
        ) = self.autocarlarewards.rewards_followlane_center_velocity_angle(
            self.centers_normal,
            self.params["current_speed"],
            self.params["target_veloc"],
            self.angle,
        )
        # print(f"\n\t{reward = }\t{done=}\t{centers_rewards_list=}")
        # input(f"\n\tin step() after rewards... waiting")

        ## -------- ... or Finish by...
        # if len(self.collision_hist) > 0:  # crashed you, baby
        #    done = True
        #    # reward = -100
        #    print(f"crash")

        self.is_finish, self.dist_to_finish = AutoCarlaUtils.finish_fix_number_target(
            self.params["location"],
            self.finish_alternate_pose,
            self.finish_pose_number,
            self.max_target_waypoint_distance,
        )
        if self.is_finish:
            print(f"Finish!!!!")
            done = True

        ########################## STATE = DQN INPUTS: (ALL NORMALIZED IN THEIR PARTICULAR RANGES)
        ## CENTERS +
        ## LINE BORDERS +
        ## V +
        ## W +
        ## ANGLE

        # print(f"\t{self.params['current_steering_angle'] =}")
        self.state = self.DQN_states_simplified_perception_normalized(
            self.params["current_speed"],
            self.params["target_veloc"],
            self.params["current_steering_angle"],
            self.angle,
            lane_centers_in_pixels,
            index_right,
            index_left,
            mask.shape[1],
        )
        # print(f"\n\t{self.state = }")

        if done:
            print(
                f"\n\t{center_reward =}, {done_center =}, {centers_rewards_list =}"
                f"\n\t{velocity_reward =}, {done_velocity =}"
                f"\n\t{heading_reward =}, {done_heading =}"
                f"\n\t{done =}, {reward =}"
                f"\n\t{self.params['current_speed'] =}, {self.params['target_veloc'] =}, {self.angle =}"
            )
            # input("step() ----> done = True")

        return self.state, reward, done, {}

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
        c = (
            self.car.car.get_control()
        )  # returns values in [-1, 1] range for throttle, steer,...(https://carla.readthedocs.io/en/0.9.13/python_api/#carla.VehicleControl)
        w = self.car.car.get_angular_velocity()  # returns in deg/sec in a 3D vector
        a = self.car.car.get_acceleration()

        ## Applied throttle, brake and steer
        curr_speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        curr_steer = math.sqrt(w.x**2 + w.y**2 + w.z**2)
        acceleration = math.sqrt(a.x**2 + a.y**2 + a.z**2)
        # print(f"in STEP() {throttle = } and {steer = }")

        throttle = self.actions[action][0]
        steer = self.actions[action][1]
        target_veloc = self.target_veloc
        brake = 0

        # self.params["Steering_angle"] = steering_angle
        self.params["location"] = (t.location.x, t.location.y)
        self.params["height"] = t.location.z

        self.params["Throttle"] = c.throttle
        self.params["Steer"] = c.steer
        self.params["Brake"] = c.brake

        self.params["current_speed"] = curr_speed
        self.params["current_steering_angle"] = curr_steer
        self.params["acceleration"] = acceleration
        self.params["target_veloc"] = target_veloc

        # Limiting Velocity up to target_speed = 30 (or any other)
        if curr_speed > target_veloc:
            throttle = 0.0

        self.car.car.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

    #####################################################################################
    # ---   methods
    #####################################################################################

    def DQN_states_simplified_perception_normalized(
        self,
        speed,
        target_veloc,
        steering,
        angle,
        lane_centers_in_pixels,
        index_right,
        index_left,
        width,
    ):

        v_normal = self.normalizing_DQN_values(speed, target_veloc, 0, is_list=True)
        w_normal = self.normalizing_DQN_values(steering, 3, 0, is_list=False)
        angle_normal = self.normalizing_DQN_values(angle, 30, 0, is_list=False)
        states_normal = self.normalizing_DQN_values(
            lane_centers_in_pixels, width, 0, is_list=True
        )
        line_borders_normal = self.normalizing_DQN_values(
            index_right + index_left, width, 0, is_list=True
        )

        DQN_states_normalized = (
            states_normal + line_borders_normal + v_normal + w_normal + angle_normal
        )

        return DQN_states_normalized

    def normalizing_DQN_values(self, values, max_value, min_value=0, is_list=True):
        # values_ = abs(values)
        # values_ = values
        values = np.array(values).reshape(-1, 1)
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        min_max_scaler.fit([[min_value], [max_value]])
        data_min_max_scaled = min_max_scaler.transform(values)
        values_normal = data_min_max_scaled.flatten().tolist()
        values_normal = [
            values_normal[i] if values_normal[i] >= 0 else -1
            for i, _ in enumerate(values_normal)
        ]

        return values_normal

    def preprocess_image(self, img):
        """
        In only Right Line mode and Sergios segmentation for Town07
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

    def preprocess_image_lane_detector(self, image):
        """
        image from lane detector with regression
        """
        ## first, we cut the upper image
        height = image.shape[0]
        image_middle_line = (height) // 2
        img_sliced = image[image_middle_line:]

        kernel = np.ones((3, 3), np.uint8)
        img_erosion = cv2.erode(img_sliced, kernel, iterations=2)

        # Convertir la imagen de BGR a RGB
        image_rgb = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2RGB)

        # Definir un rango de color  en el formato HSV
        # lower_blue = np.array([100, 50, 50])
        # upper_blue = np.array([130, 255, 255])
        # lower_red = np.array([0, 50, 50])
        lower_red = np.array([0, 200, 140])
        upper_red = np.array([0, 255, 255])
        # Definir un rango adicional para tonos cercanos al rojo
        # lower_red_additional = np.array([170, 50, 50])
        # upper_red_additional = np.array([180, 255, 255])

        # Convertir la imagen RGB a HSV
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        # Crear una máscara
        mask = cv2.inRange(image_hsv, lower_red, upper_red)

        # Crear una máscara para los píxeles rojos y cercanos al rojo
        # mask_red = cv2.inRange(image_hsv, lower_red, upper_red)
        # mask_red_additional = cv2.inRange(
        #    image_hsv, lower_red_additional, upper_red_additional
        # )

        # Unir ambas máscaras
        # mask = cv2.bitwise_or(mask_red, mask_red_additional)

        # Convertir la máscara a una máscara de tres canales
        # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Copiar la imagen original
        # imagen_rgb = np.copy(imagen_rgb)

        # Reemplazar los píxeles azules con blanco en la imagen resultante
        # image_rgb[mask == 255] = [255, 255, 255]
        # imagen_resultado[mask_rgb == 255] = [255, 255, 255]

        # Mostrar la imagen original y la imagen resultante
        # image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        # image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # cv2.imshow("RGB", image_rgb)
        # cv2.imshow("BGR", image_bgr)

        return mask

    def __preprocess_image_lane_detector(self, image):
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

        # substituting 0's by -1
        index_left = [
            index_left[x] if index_left[x] != 0 else 0 for x, _ in enumerate(index_left)
        ]

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
            ((right - left) // 2) + left if right != 0 and left != 0 else 0
            for right, left in zip(index_right, index_left)
        ]
        dist_to_center = [
            abs(centers[i] - gray_mask.shape[1] // 2) if centers[i] != 0 else 0
            for i, _ in enumerate(centers)
        ]
        dist_to_center_normalized = [
            (
                float(
                    ((gray_mask.shape[1] // 2) - centers[i]) / (gray_mask.shape[1] // 2)
                )
                if dist_to_center[i] != 0
                else 0
            )
            for i, _ in enumerate(dist_to_center)
        ]
        return (
            centers,
            dist_to_center,
            dist_to_center_normalized,
            index_right,
            index_left,
        )

    def calculate_states(
        self, width, lane_centers_in_pixels, num_regions, size_lateral_states
    ):
        # size_lateral_states = 140
        size_center_states = width - (size_lateral_states * 2)
        pixel_center_states = int(size_center_states / (num_regions - 2))

        states = [
            (
                int(((value - size_lateral_states) / pixel_center_states) + 1)
                if (width - size_lateral_states) > value > size_lateral_states
                else (
                    num_regions - 1
                    if value >= (width - size_lateral_states)
                    else num_regions
                )
            )
            for _, value in enumerate(lane_centers_in_pixels)
        ]

        ## negatives states due to 0's convert in 0
        states = [states[i] if states[i] > 0 else 0 for i, _ in enumerate(states)]

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

    def heading_car(self, central_line_points, centers_lines_points):
        """
        in general form
        central_line_points = [(image.shape[1] //2, 0), [image.shape[1] // 2, image.shape[0]]]

        central_line_points = [(320, 0), [320, 480]] for an image of 480 x 640 dimensions


        centers_lines_points is variable in size, allows different dimensions
        """
        # slope
        centers_slope = [
            np.arctan2(y2 - y1, x2 - x1)
            for (x1, y1), (x2, y2) in zip(
                centers_lines_points[:-1], centers_lines_points[1:]
            )
        ]

        # mean angle between 2 lines, central and centers
        # angulo_entre_lineas = np.abs(np.degrees(np.mean(pendientes_linea2) - np.arctan2(y2_linea1 - y1_linea1, x2_linea1 - x1_linea1)))
        angle = np.abs(
            np.degrees(
                np.mean(centers_slope)
                - np.arctan2(
                    central_line_points[1][1] - central_line_points[0][1],
                    central_line_points[1][0] - central_line_points[0][0],
                )
            )
        )

        return angle

    ##################
    #
    # ONLY RIGHT LANE
    ####################################################################################

    def step_only_right_line(self, action):
        """
        state: sp
        actions: discretes
                Only Right Line
        """

        self.control_discrete_actions(action)

        ########### --- calculating STATES
        mask = self.preprocess_image(self.sensor_camera_red_mask.front_red_mask_camera)
        AutoCarlaUtils.show_image(
            "mask",
            mask,
            300,
            10,
        )

        ########### --- Calculating center ONLY with right line
        (
            errors,
            dist_to_center,
            dist_to_center_normalized,
            index_right,
            ground_truth_normal_values,
        ) = self.calculate_lane_centers_with_only_right_line(mask)

        print(
            f"\n\t{errors = }\n\t{dist_to_center =}\n{dist_to_center_normalized =}\n\t{index_right = }\n\t{ground_truth_normal_values =}"
        )

        ## STATES
        (
            self.states,
            drawing_lines_states,
            drawing_numbers_states,
        ) = self.calculate_states(
            mask.shape[1],
            dist_to_center,
            self.num_regions,
            size_lateral_states=140,
        )
        print(
            f"\n\t{self.states = }\n\t{drawing_lines_states =}\n\t{drawing_numbers_states =}"
        )

        index_left = [0 for i, _ in enumerate(index_right)]  # only for drawing
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
            dist_to_center,
            drawing_lines_states,
            drawing_numbers_states,
        )

        ## -------- Ending Step()...
        ###################################
        #
        #       REWARDS
        ###################################

        done = False
        reward, done = self.autocarlarewards.rewards_sigmoid_only_right_line(
            self.dist_normalized, ground_truth_normal_values, self.params
        )

        return self.states, reward, done, {}

    def calculate_right_line(self, mask, x_row):
        """
        All below code has been merged into calculate_lane_centers_with_only_right_line() method
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

    def calculate_lane_centers_with_only_right_line(self, mask):
        ## get total lines in every line point
        lines = [mask[self.x_row[i], :] for i, _ in enumerate(self.x_row)]

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

        image_center = mask.shape[1] // 2

        # TODO: OJO HAY Q DISTINGUIR + Y - DEL CENTRO
        dist_to_center = [
            image_center - index_right[i] for i, _ in enumerate(index_right)
        ]

        # TODO: VERIFICAR ESTE VALOR
        dist_to_center_normalized = [
            float((image_center - index_right[i]) / image_center)
            for i, _ in enumerate(index_right)
        ]

        ground_truth_normal_values = [
            correct_normalized_distance[value] for i, value in enumerate(self.x_row)
        ]

        errors = [
            dist_to_center_normalized[index] - ground_truth_normal_values[index]
            for index, _ in enumerate(dist_to_center_normalized)
        ]

        return (
            errors,
            dist_to_center,
            dist_to_center_normalized,
            index_right,
            ground_truth_normal_values,
        )
