from cmath import cos
import math
import cv2
from cv2 import CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gym import spaces
from gym.utils import seeding
from sensor_msgs.msg import Image
from datetime import datetime
import time
from statistics import mean
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from PIL import Image as im
from agents.utils import print_messages

from rl_studio.envs.gazebo.autoparking.image_f1 import ImageF1
from rl_studio.envs.gazebo.autoparking.models.autoparking_env import AutoparkingEnv
from rl_studio.envs.gazebo_utils import set_new_pose


class DDPGAutoparkingEnvGazebo(AutoparkingEnv):
    def __init__(self, **config):

        ParkingcarEnv.__init__(self, **config)
        self.image = ImageF1()
        # self.cv_image_pub = None
        self.image_raw_from_topic = None
        self.f1_image_camera = None

        self.sensor = config["sensor"]

        # Image
        self.image_resizing = config["image_resizing"] / 100
        self.new_image_size = config["new_image_size"]
        self.raw_image = config["raw_image"]
        self.height = int(config["height_image"] * self.image_resizing)
        # self.height = int(config['height_image'])
        self.width = int(config["width_image"] * self.image_resizing)
        # self.width = int(config['width_image'])
        self.center_image = int(config["center_image"] * self.image_resizing)
        # self.center_image = int(config['center_image'])
        self.image_middle_line = self.height // 2
        # self.num_regions = config["num_regions"]
        # self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config["telemetry_mask"]
        # self.poi = config["x_row"][0]
        self.image_center = None

        # States
        self.state_space = config["state_space"]
        self.states = config["states"]
        print(self.state_space)
        if self.state_space == "sp_curb":
            self.poi = self.states["poi"]
            self.regions = self.states["regions"]
            self.pixels_cropping = self.states["pixels_cropping"]
        else:
            self.x_row = config["x_row"]

        # Actions
        # self.beta_1 = -(config["actions"]['w_left'] / (config["actions"]['v_max'] - config["actions"]['v_min']))
        # self.beta_0 = -(self.beta_1 * config["actions"]['v_max'])
        self.action_space = config["action_space"]
        self.actions = config["actions"]

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]
        # self.beta_1 = config["beta_1"]
        # self.beta_0 = config["beta_0"]

        # Others
        self.telemetry = config["telemetry"]

    #####################################################################################################

    def show_image(self, name, img, waitkey):
        window_name = f"{name}"
        img_centroid = img

        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    #######################################################################################

    def image_msg_to_image(self, img, cv_image):
        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image

    def get_camera_info(self):
        image_data = None
        f1_image_camera = None
        success = False

        while image_data is None or success is False:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            if np.any(cv_image):
                success = True

        return f1_image_camera, cv_image

    #####################################################################################################
    ## STATES

    def state_sp_curb_processing(self, image, poi, pixels_cropping, num_regions):
        height, width, _ = image.shape
        middle_height = height // 2
        center_image = width // 2
        # quartil = 200

        # cutting image (240, 400)
        img_cropping = image[
            middle_height:,
            center_image - pixels_cropping : center_image + pixels_cropping,
        ]

        # convert to B&W
        img_gray = cv2.cvtColor(img_cropping, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=2)

        # get columns
        new_height, new_width, _ = mask.shape
        # reference_columns = [1, new_width // 2, new_width - 1]
        reference_columns = [
            int(((new_width / poi) // 2) + (i * new_width / poi)) for i in range(poi)
        ]
        columns = [
            mask[:, reference_columns[idx], 0]
            for idx, _ in enumerate(reference_columns)
        ]

        # get distance from every column
        distances2bottom_pixels = []
        for idx, _ in enumerate(reference_columns):
            try:
                distances2bottom_pixels.append(
                    new_height - np.min(np.nonzero(columns[idx]))
                )
            except:
                distances2bottom_pixels.append(-100)

        # dist2bottom_normalized = [
        #        float((new_height - distances2bottom_pixels[i]) / new_height)
        #        for i, _ in enumerate(distances2bottom_pixels)
        # ]

        # calculate points in regions
        regions = [
            int((new_height - x) / num_regions) + 1 if x > 0 else 0
            for _, x in enumerate(distances2bottom_pixels)
        ]

        return regions, distances2bottom_pixels

    def image_resizing(self, image, middle_height, width_quartil):
        """
        no me funciona yet
        """
        height, width, _ = image.shape
        # middle_height = height // 2
        center_image = width // 2
        # quartil = 200
        img_resized = image[
            int(middle_height) :,
            center_image - int(width_quartil) : center_image + int(width_quartil),
        ]
        return img_resized

    def state_image_processing(self, image, size):
        height, width, _ = image.shape
        # middle_height = height // 2
        middle_height = 280
        center_image = width // 2
        quartil = 200

        # cutting image
        state_img = image[
            middle_height:, center_image - quartil : center_image + quartil
        ]
        # self.show_image("state_img", state_img, 100)

        # resizing image
        img_resized = cv2.resize(state_img, (size[0], size[1]), cv2.INTER_AREA)
        # img_resized = cv2.resize(image, (size[0], size[1]), cv2.INTER_AREA)
        # self.show_image("state_img_resized", img_resized, 100)
        return img_resized

    def image_preprocessing(self, img, type="b&w"):
        # img_sliced = img[self.image_middle_line :]
        img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_proc, 200, 255, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=2)

        return mask

    #####################################################################################################
    def curb_image_processing(self, image):
        height, width, _ = image.shape
        # middle_height = height // 2
        middle_height = 280
        center_image = width // 2
        quartil = 50

        # resizing image
        mask_curb = image[
            middle_height:, center_image - quartil : center_image + quartil
        ]
        self.show_image("mask_curb 240x100", mask_curb, 100)
        new_height, new_width, _ = mask_curb.shape

        # columns with blanks: 1, center, width - 1
        reference_columns = [1, new_width // 2, new_width - 1]

        columns = [
            mask_curb[:, reference_columns[idx], 0]
            for idx, _ in enumerate(reference_columns)
        ]

        # print_messages(
        #    "en curb_image_processing()",
        #    image_shape=image.shape,
        #    mask_curb_shape=mask_curb.shape,
        # )

        return reference_columns, columns, new_height

    def curb_distances_to_bottom(self, reference_columns, columns, height):
        distances2bottom_pixels = []
        for idx, _ in enumerate(reference_columns):
            try:
                distances2bottom_pixels.append(
                    height - np.min(np.nonzero(columns[idx]))
                )
            except:
                distances2bottom_pixels.append(-100)

        # print(f"en curb_distances_to_bottom(): distances2bottom = {distances2bottom}")

        # convert pixels to [0,1]
        # diff = height - distances2bottom[1]
        # dist2bottom = (diff - distances2bottom[1]) / (diff + distances2bottom[1])
        # dist2bottom = float(height - distances2bottom) / float(height)
        dist2bottom_normalized = [
            float((height - distances2bottom_pixels[i]) / height)
            for i, _ in enumerate(distances2bottom_pixels)
        ]

        return dist2bottom_normalized, distances2bottom_pixels

    def curb_angle(self, distances_bottom, reference_columns):
        a = distances_bottom[2] - distances_bottom[0]
        b = reference_columns[2] - reference_columns[0]
        hipt = np.hypot(a, b)
        angle_in_degrees = math.degrees(
            np.arccos((b**2 + hipt**2 - a**2) / (2 * b * hipt))
        )
        angle_in_radians = np.arccos((b**2 + hipt**2 - a**2) / (2 * b * hipt))
        return angle_in_degrees, angle_in_radians

    def lines_image_preprocessing(self, image):
        height, width, _ = image.shape
        middle_height = height // 2
        # middle_height = 280
        center_image = width // 2
        quartil = 200

        # resizing image
        mask_lines = image[
            middle_height:, center_image - quartil : center_image + quartil
        ]
        new_height, new_width, _ = mask_lines.shape
        new_center_image = new_width // 2

        self.show_image("mask_lines 240x400", mask_lines, 100)
        # self.show_image(
        #    "mask_lines 240x200 right side", mask_lines[:, new_center_image + 1 :], 1
        # )

        lines_row = [middle_height - i for i in range(5, middle_height, 40)]

        lines_left = [
            mask_lines[lines_row[idx], :new_center_image]
            for idx, x in enumerate(lines_row)
        ]

        lines_right = [
            mask_lines[lines_row[idx], new_center_image:, 0]
            for idx, x in enumerate(lines_row)
        ]

        # print_messages(
        #    "en lines_image_preprocessing()",
        #    image_shape=image.shape,
        #    mask_lines_shape=mask_lines.shape,
        #    lines_row=lines_row,
        #    new_center_image=new_center_image,
        # lines_left_0=lines_left[0],
        # lines_right_0=lines_right[0],
        # )

        return lines_row, lines_left, lines_right, new_center_image

    def distances_right_line_to_center(self, lines_right, lines_row):
        distancias2right_lane = []
        for i, _ in enumerate(lines_row):
            try:
                distancias2right_lane.append(np.min(np.nonzero(lines_right[i])))
            except:
                distancias2right_lane.append(0)

        return distancias2right_lane

    def distances_left_line_to_center(self, lines_left, lines_row, center_image):
        distancias2left_lane = []
        for i, _ in enumerate(lines_row):
            try:
                distancias2left_lane.append(
                    center_image - np.max(np.nonzero(lines_left[i]))
                )
            except:
                distancias2left_lane.append(0)

        return distancias2left_lane

    def ratio_left2right_lines(self, distances2right_line, distances2left_line):
        # ratios = [
        #    abs((x - y) / (x + y))
        #    for x, y in zip(distances2right_line, distances2left_line)
        # ]
        ratios = []
        for x, y in zip(distances2right_line, distances2left_line):
            try:
                ratios.append(abs((x - y) / (x + y)))
            except:
                ratios.append(0)

        avg = [ratios[x] for x in range(len(ratios)) if ratios[x] > 0]
        try:
            return mean(avg)
        except:
            return 0

    def params_parkingslot(self, img):
        # image = self.image_preprocessing(img, (img.shape[0], img.shape[1]))
        image = img

        # preprocessing image for curb distance to bottom
        reference_columns, columns, curb_new_height = self.curb_image_processing(
            self.image_preprocessing(img, "b&w")
        )

        # curb distance to bottom
        distances2bottom_norm, distances2bottom_pixels = self.curb_distances_to_bottom(
            reference_columns, columns, curb_new_height
        )

        # curb angle
        angle_in_degrees, angle_in_radians = self.curb_angle(
            distances2bottom_pixels, reference_columns
        )

        # preprocessing image for left-right lines
        (
            lines_row,
            lines_left,
            lines_right,
            new_center_image,
        ) = self.lines_image_preprocessing(self.image_preprocessing(image, "b&w"))

        # distances to right line
        distances2right_line = self.distances_right_line_to_center(
            lines_right, lines_row
        )

        # distances to left line
        distances2left_line = self.distances_left_line_to_center(
            lines_left, lines_row, new_center_image
        )

        # calculating ratio between left and right line
        ratio_l_r = self.ratio_left2right_lines(
            distances2right_line, distances2left_line
        )

        return (
            distances2bottom_norm,
            distances2bottom_pixels,
            angle_in_degrees,
            angle_in_radians,
            distances2right_line,
            distances2left_line,
            ratio_l_r,
        )

    #####################################################################################################

    """
    def rewards_near_parkingspot_discrete(self, curb_dist, ratio_l_r):
        if 1.0 > curb_dist >= 0.87:
            if 0 <= ratio_l_r <= 0.1:
                reward = 80
            elif 0.1 < ratio_l_r <= 0.3:
                reward = 70
            elif 0.3 < ratio_l_r <= 1:
                reward = 40

        elif 0.87 > curb_dist >= 0.7:
            if 0 <= ratio_l_r <= 0.1:
                reward = 69
            elif 0.1 < ratio_l_r <= 0.3:
                reward = 50
            elif 0.3 < ratio_l_r <= 0.5:
                reward = 20
            elif 0.5 < ratio_l_r <= 1:
                reward = 3

        elif 0.7 > curb_dist >= 0.4:
            if 0 <= ratio_l_r <= 0.1:
                reward = 30
            elif 0.1 < ratio_l_r <= 0.3:
                reward = 15
            elif 0.3 < ratio_l_r <= 0.5:
                reward = 10
            elif 0.5 < ratio_l_r <= 1:
                reward = 2

        elif 0.4 > curb_dist >= 0.0:
            if 0 <= ratio_l_r <= 0.1:
                reward = 19
            elif 0.1 < ratio_l_r <= 0.3:
                reward = 9
            elif 0.3 < ratio_l_r <= 0.5:
                reward = 5
            elif 0.5 < ratio_l_r <= 1:
                reward = 1

        return reward
        """

    def rewards_near_parkingspot_v2dist_2opcion(self, curb_dist, v, angle, ratio_l_r):
        """produce 0's. De momento la desechamos"""
        vmax = 3.0
        vmin = -1
        dmax = 0.85
        dmin = 0
        B_1 = vmax / (dmax - dmin)
        B_0 = vmax

        v_target = B_0 - (B_1 * curb_dist[1])
        error = abs(v - v_target)
        # try:
        #    reward_error = 1 / error
        # except:
        #    reward_error = 200

        try:
            reward_near = (1 / error) * math.cos(angle) * (1 / ratio_l_r)
        except:
            reward_near = 20000

        return reward_near

    def rewards_near_parkingspot_v2dist(self, curb_dist, v, angle, ratio_l_r):
        vmax = 3.0
        vmin = -1
        dmax = 0.85
        dmin = 0
        B_1 = vmax / (dmax - dmin)
        B_0 = vmax

        v_target = B_0 - (B_1 * curb_dist[1])
        error = abs(v - v_target)
        try:
            reward_near = (
                (1 / math.exp(error)) * math.cos(angle) * (1 / math.exp(ratio_l_r))
            )
        except:
            reward_near = 0

        return reward_near

    def rewards_near_parkingspot_discrete(
        self, curb_dist, v, angle=None, ratio_l_r=None
    ):
        if curb_dist < 0.5 and v > 1:
            reward = 10
        elif (0.84 > curb_dist >= 0.5) and (1 >= v > 0.5):
            reward = 20
        elif (0.95 > curb_dist >= 0.84) and (0.5 >= v > 0):
            reward = 40
        else:
            reward = 1

        return reward

    def rewards_near_parkingspot(self, curb_dist, v, angle, ratio_l_r):
        # v_0_count = 0
        # penalty_reward = -100
        # goal_reward = 10_000

        # WRONG v = 0 and far from curb or overpassing curb
        if v == 0 and (curb_dist[1] < 0.6 or mean(curb_dist) >= 1):
            done = True
            reward = self.rewards["penalty_reward"]
            # info = "out for v = 0 and far from curb or overpassing curb"
            print_messages(
                "rewards: done for v = 0 and far from curb or overpassing curb",
            )
        # WRONG left-right ratio unbalanced
        elif curb_dist[1] < 0.7 and ratio_l_r > 0.9:
            done = True
            reward = self.rewards["penalty_reward"]
            # info = "out left-right ratio unbalanced"
            print_messages(
                "rewards: done left-right ratio unbalanced",
            )
        # WRONG curb angle very inclined
        elif angle > 10:
            done = True
            reward = self.rewards["penalty_reward"]
            # info = "curb angle very inclined"
            print_messages(
                "rewards: done angle very inclined",
            )
        # WRONG out of margin
        elif curb_dist[1] > 0.83 and v != 0:
            done = True
            reward = self.rewards["penalty_reward"]

        # done for distance
        elif 1 > curb_dist[1] > 0.83 and v == 0:
            done = True
            reward = self.rewards["goal_reward"]
            print_messages("rewards: GOAAAAALLL!!")

        # we must avoid v=0 for long time
        # elif v == 0 and (curb_dist[1] < 0.92 or curb_dist != -100):
        #    v_0_count += 1
        #    if v_0_count > 3:
        #        done = True
        #        reward = penalty_reward
        #    print_messages(
        #        "rewards: out for v=0 for long time",
        #    )

        # everything ok
        else:
            reward = self.rewards_near_parkingspot_discrete(
                curb_dist[1],
                v,
                angle,
                ratio_l_r,
            )
            # reward = self.rewards_near_parkingspot_discrete(
            #    curb_dist[1],
            #    ratio_l_r,
            # )
            done = False
            # info = "ok"

        return round(reward, 2), done

    #####################################################################################################
    # Rewards Simplified percepction

    #####################################################################################################

    def rewards_near_parkingspot_linear_sp(self, curb_dist, v, angle, ratio_l_r):
        vmax = 3.0
        vmin = -1
        dmax = 0.85
        dmin = 0
        B_1 = vmax / (dmax - dmin)
        B_0 = vmax
        # print(curb_dist)
        v_target = B_0 - (B_1 * curb_dist)
        error = abs(v - v_target)
        try:
            reward_near = (
                (1 / math.exp(error)) * math.cos(angle) * (1 / math.exp(ratio_l_r))
            )
        except:
            reward_near = 0

        # discretizing rewards for fast speed learning
        if 1 >= reward_near >= 0.9:
            reward = 1
        elif 0.9 > reward_near >= 0.75:
            reward = 0.5
        elif 0.75 > reward_near >= 0.5:
            reward = 0.2
        else:
            reward = 0

        return reward

    def rewards_near_parkingspot_discrete_sp(self, curb_dist, v):
        if curb_dist < 0.5 and v > 1:
            reward = 5
        elif (0.84 > curb_dist >= 0.5) and (1 >= v > 0.5):
            reward = 10
        # elif (0.95 > curb_dist >= 0.84) and (0.5 >= v > 0):
        #    reward = 40
        else:
            reward = 0

        # if (0.84 > curb_dist >= 0.5) and v > 1:
        #    reward = 20
        # elif (0.95 > curb_dist >= 0.84):
        #    reward = 40
        # else:
        #    reward = 1

        return reward

    def rewards_near_parkingspot_sp(self, curb_dist, v, angle, ratio_l_r):
        # v_0_count = 0
        # penalty_reward = -100
        # goal_reward = 10_000

        # WRONG out of margin
        if curb_dist[1] > 0.83 and v != 0:
            done = True
            reward = self.rewards["penal_reward"]

        # done for distance
        elif 1 > curb_dist[1] > 0.83 and v == 0:
            done = True
            reward = self.rewards["goal_reward"]
            print_messages("rewards: GOAAAAALLL!!")

        # everything ok
        else:
            reward = self.rewards_near_parkingspot_discrete_sp(
                curb_dist[1],
                v,
                angle,
                ratio_l_r,
            )
            done = False

        return round(reward, 2), done

    #####################################################################################################
    # LASER
    ############################################

    @staticmethod
    def discrete_observation(data, new_ranges):

        discrete_ranges = []
        min_range = 0.05
        done = False
        mod = len(data.ranges) / new_ranges
        filter_data = data.ranges[10:-10]
        for i, item in enumerate(filter_data):
            if i % mod == 0:
                if filter_data[i] == float("Inf") or np.isinf(filter_data[i]):
                    discrete_ranges.append(6)
                elif np.isnan(filter_data[i]):
                    discrete_ranges.append(0)
                else:
                    discrete_ranges.append(int(filter_data[i]))
            if min_range > filter_data[i] > 0:
                print("Data ranges: {}".format(data.ranges[i]))
                done = True
                break

        return discrete_ranges, done

    def get_laser_info(self):
        laser_data = None
        success = False
        while laser_data is None or not success:
            try:
                laser_data = rospy.wait_for_message(
                    "/F1ROS/laserC/scan", LaserScan, timeout=5
                )
            finally:
                success = True

        return laser_data

    #####################################################################################################

    def reset(self):
        if self.sensor == "camera":
            return self.reset_camera()
        elif self.sensor == "laser":
            return self.reset_laser()

    def reset_laser(self):
        self._gazebo_set_new_pose()
        self._gazebo_reset()
        self._gazebo_unpause()

        # Read laser data
        # laser_data = None
        # success = False
        # while laser_data is None or not success:
        #    try:
        #        laser_data = rospy.wait_for_message(
        #            "/F1ROS/laserC/scan", LaserScan, timeout=5
        #        )
        #    finally:
        #        success = True

        laser_data = self.get_laser_info()
        self._gazebo_pause()

        state = self.discrete_laser_beans_mins(laser_data, 5)
        range_center = self.max_laser_range_ahead(laser_data)
        print_messages(
            "en reset_laser()",
            state=state,
            laser_data=laser_data,
            len_data_ranges=len(laser_data.ranges),
            range_center=range_center,
        )
        return state

    ###############

    def reset_camera(self):

        if self.alternate_pose:
            # print(f"\n[INFO] ===> Necesary implement self._gazebo_set_new_pose()...class F1DDPGCameraEnv(F1Env) -> def reset_camera() \n")
            self._gazebo_set_random_new_pose()  # Mine, it works fine!!!
            # pos_number = set_new_pose(self.circuit_positions_set) #not using. Just for V1.1
        else:
            self._gazebo_set_new_pose()
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        f1_image_camera, _ = self.get_camera_info()
        # self.show_image("f1_image_camera in reset()", f1_image_camera.data, 3)

        self._gazebo_pause()

        # Calculating STATE
        # state_img_processing = self.state_image_processing(
        #    f1_image_camera.data, (self.new_image_size, self.new_image_size)
        # )
        # print(state_img_preprocessing.shape)
        # state = np.array(
        #    self.image_preprocessing(
        #        state_img_processing,
        #        "b&w",
        #    )
        # )

        # Calculating STATE
        # we want state as image preprocessed
        if self.state_space == "image":
            state_img_processed = self.state_image_processing(
                f1_image_camera.data, (self.new_image_size, self.new_image_size)
            )
            # print(state_img_preprocessing.shape)
            self.show_image(
                "state_img_mask",
                self.image_preprocessing(
                    state_img_processed,
                    "b&w",
                ),
                100,
            )
            state = np.array(
                self.image_preprocessing(
                    state_img_processed,
                    "b&w",
                )
            )
            state_size = state.shape
        elif self.state_space == "sp_curb":
            state, _ = self.state_sp_curb_processing(
                f1_image_camera.data, self.poi, self.pixels_cropping, self.regions
            )
            state_size = len(state)
        return state, state_size

    #####################################################################################################

    def step(self, action):

        self._gazebo_unpause()
        vel_cmd = Twist()

        # get velocity linear and angular
        if self.action_space == "continuous":
            vel_cmd.linear.x = action[0][0]
            vel_cmd.angular.z = action[0][1]
        else:
            vel_cmd.linear.x = self.actions[action][0]
            vel_cmd.angular.z = self.actions[action][1]

        # publishing to Topic
        self.vel_pub.publish(vel_cmd)

        # Get camera info (original 480x640)
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()

        # Calculating STATE
        # we want state as image preprocessed
        if self.state_space == "image":
            state_img_processed = self.state_image_processing(
                f1_image_camera.data, (self.new_image_size, self.new_image_size)
            )
            # print(state_img_preprocessing.shape)
            self.show_image(
                "state_img_mask",
                self.image_preprocessing(
                    state_img_processed,
                    "b&w",
                ),
                100,
            )
            state = np.array(
                self.image_preprocessing(
                    state_img_processed,
                    "b&w",
                )
            )
        elif self.state_space == "sp_curb":
            state, curb_distances2bottom_pixels_sp = self.state_sp_curb_processing(
                f1_image_camera.data, self.poi, self.pixels_cropping, self.regions
            )

        elif self.state_space == "sp_curbandlines":
            state = self.state_sp_curbandlines_processing(
                f1_image_camera.data, self.poi, self.pixels_cropping, self.regions
            )

        # vamos a probar con la imagen STATE pero primero la convertimos a B&N y luego resizing
        # state_img_ = self.image_preprocessing(f1_image_camera.data)
        # state_ = self.state_image_processing(
        #    state_img_, (self.new_image_size, self.new_image_size)
        # )
        # self.show_image(
        #    "state_img_mask_alreves",
        #    state_,
        #    100,
        # )

        # otra manera de imagen
        # state_image_B&W = self.image_resizing(f1_image_camera.data, 300, 200)
        # m = 280
        # w = 200
        # state_image_bw = self.image_preprocessing(f1_image_camera.data)
        # state_img_oneresize = self.image_resizing(state_image_bw, m, w)
        # state_img_secondresize = self.state_image_processing(
        #    state_img_oneresize, (self.new_image_size, self.new_image_size)
        # )
        # self.show_image(
        #    "state_img_secondresize",
        #    state_img_secondresize,
        #    100,
        # )

        # calculating params parking lot
        (
            curb_distance2bottom_norm,
            curb_distance2bottom_pixels,
            curb_angle_degrees,
            curb_angle_in_radians,
            distances2right_line,
            distances2left_line,
            ratio_left2right_lines,
        ) = self.params_parkingslot(f1_image_camera.data)

        done = False
        ### reward & done
        reward, done = self.rewards_near_parkingspot_sp(
            curb_distance2bottom_norm,
            vel_cmd.linear.x,
            curb_angle_in_radians,
            ratio_left2right_lines,
        )

        # info = curb_distance2bottom_norm[1]

        print_messages(
            "in step()",
            # f1_image_camera_data=f1_image_camera.data.shape,
            v=vel_cmd.linear.x,
            w=vel_cmd.linear.z,
            # state_dims=state.shape,
            state=state,
            curb_distances2bottom_pixels_sp=curb_distances2bottom_pixels_sp,
            curb_distance2bottom_norm=curb_distance2bottom_norm,
            curb_distance2bottom_pixels=curb_distance2bottom_pixels,
            curb_angle_degrees=curb_angle_degrees,
            curb_angle_in_radians=curb_angle_in_radians,
            distances2right_line=distances2right_line,
            distances2left_line=distances2left_line,
            ratio_left2right_lines=ratio_left2right_lines,
            reward=reward,
            done=done,
        )
        return state, reward, done, {}
