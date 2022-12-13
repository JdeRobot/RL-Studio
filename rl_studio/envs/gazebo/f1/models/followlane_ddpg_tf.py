"""
- Task: Follow Lane 
- Algorithm: DDPG
- actions: discrete and continuous
- State: Simplified perception

"""

import math

from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np

# from PIL import Image as im

import rospy
from sensor_msgs.msg import Image

# from sklearn.cluster import KMeans
# from sklearn.utils import shuffle

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.gazebo.f1.image_f1 import ImageF1
from rl_studio.envs.gazebo.f1.models.camera import F1GazeboCamera
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env
from rl_studio.envs.gazebo.f1.models.utils import F1GazeboUtils
from rl_studio.envs.gazebo.f1.models.rewards import (
    rewards_discrete_follow_lane,
    rewards_discrete,
    reward_v_center_step,
    rewards_discrete_follow_right_lane,
    reward_v_w_center_linear_no_working_at_all,
    reward_v_w_center_linear_second,
    reward_v_w_center_linear_first_formula,
)


class FollowLaneDDPGF1GazeboTF(F1Env):
    def __init__(self, **config):

        F1Env.__init__(self, **config)
        self.image = ImageF1()
        self.image_raw_from_topic = None
        self.f1_image_camera = None
        self.sensor = config["sensor"]

        # Image
        self.image_resizing = config["image_resizing"] / 100
        self.new_image_size = config["new_image_size"]
        self.raw_image = config["raw_image"]
        self.height = int(config["height_image"] * self.image_resizing)
        self.width = int(config["width_image"] * self.image_resizing)
        self.center_image = int(config["center_image"] * self.image_resizing)
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config["telemetry_mask"]
        self.poi = config["x_row"][0]
        self.image_center = None
        self.right_lane_center_image = config["center_image"] + (
            config["center_image"] // 2
        )
        self.lower_limit = config["lower_limit"]

        # States
        self.state_space = config["states"]
        if self.state_space == "spn":
            self.x_row = [i for i in range(1, int(self.height / 2) - 1)]
        else:
            self.x_row = config["x_row"]

        # Actions
        self.action_space = config["action_space"]
        self.actions = config["actions"]

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]
        self.beta_1 = self.actions["w"][1] / (
            self.actions["v"][1] - self.actions["v"][0]
        )
        self.beta_0 = self.beta_1 * self.actions["v"][1]

        # Others
        self.telemetry = config["telemetry"]

        print_messages(
            "FollowLaneDDPGF1GazeboTF() PROVISIONAL",
            actions=self.actions,
            len_actions=len(self.actions),
            actions_v=self.actions["v"],
            actions_w=self.actions["w"],
            beta_1=self.beta_1,
            beta_0=self.beta_0,
        )

    #################################################################################
    # reset
    #################################################################################

    def reset(self):
        """
        Main reset. Depending of:
        - sensor
        - states: images or simplified perception (sp)

        """
        # ic(sensor)
        if self.sensor == "camera":
            return self.reset_camera()

    def reset_camera(self):
        self._gazebo_reset()
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_follow_rigth_lane()
        else:
            self._gazebo_set_fix_pose_f1_follow_right_lane()

        self._gazebo_unpause()
        # get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()
        # calculating State
        # If image as observation
        if self.state_space == "image":
            state = np.array(
                F1GazeboCamera.preprocessing_black_white_32x32(f1_image_camera.data)
            )
            state_size = state.shape

        # simplified perception as observation
        else:
            centrals_in_pixels, centrals_normalized = self.calculate_centrals_lane(
                f1_image_camera.data
            )
            states = self.calculate_observation(centrals_in_pixels)
            state = [states[0]]
            state_size = len(state)

        return state, state_size

    #################################################################################
    # Camera
    #################################################################################
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
            # f1_image_camera = image_msg_to_image(image_data, cv_image)
            if np.any(cv_image):
                success = True

        return f1_image_camera, cv_image

    def image_msg_to_image(self, img, cv_image):
        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image

    #################################################################################
    # Center
    #################################################################################

    def processed_image(self, img):
        """
        Convert img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [mask[self.x_row[idx], :] for idx, x in enumerate(self.x_row)]
        centrals = list(map(self.get_center, lines))

        centrals_normalized = [
            float(self.center_image - x) / (float(self.width) // 2)
            for _, x in enumerate(centrals)
        ]
        # print_messages(
        #    "",
        #    lines=lines,
        #    centrals=centrals,
        # )
        self.show_image_with_centrals(
            "centrals", mask, 5, int(centrals[0]), centrals_normalized
        )

        return centrals, centrals_normalized

    @staticmethod
    def get_center(lines):
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            return 0

    def calculate_observation(self, state: list) -> list:
        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.center_image - x) / self.pixel_region) + 1)

        return final_state

    def calculate_centrals_lane(self, img):
        image_middle_line = self.height // 2
        # cropped image from second half to bottom line
        img_sliced = img[image_middle_line:]
        # convert to black and white mask
        # lower_grey = np.array([30, 32, 22])
        # upper_grey = np.array([128, 128, 128])
        img_gray = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
        # get Lines to work for
        lines = [mask[self.x_row[idx], :] for idx, _ in enumerate(self.x_row)]
        # added last line (239), to control center line in bottom
        lines.append(mask[self.lower_limit, :])

        centrals_in_pixels = list(map(self.get_center_right_lane, lines))
        centrals_normalized = [
            abs(float(self.center_image - x) / (float(self.width) // 2))
            for _, x in enumerate(centrals_in_pixels)
        ]

        F1GazeboUtils.show_image_with_centrals(
            "mask", mask, 5, centrals_in_pixels, centrals_normalized, self.x_row
        )

        return centrals_in_pixels, centrals_normalized

    @staticmethod
    def get_center_right_lane(lines):
        try:
            # inversed line
            inversed_lane = [x for x in reversed(lines)]
            # cut off right blanks
            inv_index_right = np.argmin(inversed_lane)
            # cropped right blanks
            cropped_lane = inversed_lane[inv_index_right:]
            # cut off central line
            inv_index_left = np.argmax(cropped_lane)
            # get real lane index
            index_real_right = len(lines) - inv_index_right
            if inv_index_left == 0:
                index_real_left = 0
            else:
                index_real_left = len(lines) - inv_index_right - inv_index_left
            # get center lane
            center = (index_real_right - index_real_left) // 2
            center_lane = center + index_real_left

            # avoid finish line or other blank marks on the road
            if center_lane == 0:
                center_lane = 320

            return center_lane

        except ValueError:
            return 0

    @staticmethod
    def get_center_circuit_no_wall(lines):
        try:
            pos_final_linea_negra = np.argmin(lines) + 15
            carril_derecho_entero = lines[pos_final_linea_negra:]
            final_carril_derecho = np.argmin(carril_derecho_entero)
            lim_izq = pos_final_linea_negra
            lim_der = pos_final_linea_negra + final_carril_derecho

            punto_central_carril = (lim_der - lim_izq) // 2
            punto_central_absoluto = lim_izq + punto_central_carril
            return punto_central_absoluto

        except ValueError:
            return 0

    #################################################################################
    # step
    #################################################################################

    def step(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()

        if self.action_space == "continuous":
            vel_cmd.linear.x = action[0][0]
            vel_cmd.angular.z = action[0][1]
        else:
            vel_cmd.linear.x = self.actions[action][0]
            vel_cmd.angular.z = self.actions[action][1]

        self.vel_pub.publish(vel_cmd)
        # get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()

        ######### center
        points, centrals_normalized = self.processed_image(f1_image_camera.data)
        if self.state_space == "spn":
            self.point = points[self.poi]
        else:
            self.point = points[0]
        # center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        center = float(self.center_image - self.point) / (float(self.width) // 2)

        # print_messages(
        #   "step()",
        #   points=points,
        #   points_0=points[0],
        #   center=center,
        #   self_x_row=self.x_row,
        #   centrals_normalized=centrals_normalized,
        # )
        self.show_image_with_centrals(
            "centrals",
            f1_image_camera.data[self.height // 2 :],
            5,
            points[0],
            centrals_normalized,
        )
        ########## calculating State
        # If image as observation
        if self.state_space == "image":
            state = np.array(
                self.image_preprocessing_black_white_32x32(f1_image_camera.data)
            )

        # simplified perception as observation
        else:
            state = self.calculate_observation(points)

        ########## calculating Rewards
        if self.reward_function == "linear":
            reward, done = self.reward_v_center_step(vel_cmd, center, step)
        elif self.reward_function == "linear_follow_line":
            reward, done = self.reward_v_w_center_linear(vel_cmd, center)
        else:
            reward, done = self.rewards_discrete_follow_lane(center)

        # print_messages(
        #    "in step()",
        #    self_x_row=self.x_row,
        #    points=points,
        #    center=center,
        #    reward=reward,
        #    done=done,
        # state=state,
        # )

        return state, reward, done, {}
