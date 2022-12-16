#############################################
# - Task: Follow Lane
# - Algorithm: DDPG
# - actions: discrete and continuous
# - State: Simplified perception
#
############################################

import math

from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np

import rospy
from sensor_msgs.msg import Image

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.gazebo.f1.image_f1 import ImageF1
from rl_studio.envs.gazebo.f1.models.images import F1GazeboImages
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env
from rl_studio.envs.gazebo.f1.models.utils import F1GazeboUtils
from rl_studio.envs.gazebo.f1.models.rewards import F1GazeboRewards
from rl_studio.envs.gazebo.f1.models.simplified_perception import (
    F1GazeboSimplifiedPerception,
)


class FollowLaneDDPGF1GazeboTF(F1Env):
    def __init__(self, **config):

        F1Env.__init__(self, **config)
        self.simplifiedperception = F1GazeboSimplifiedPerception()
        self.f1gazeborewards = F1GazeboRewards()
        self.f1gazeboutils = F1GazeboUtils()
        self.f1gazeboimages = F1GazeboImages()

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
        if self.action_space == "continuous":
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
            # actions_v=self.actions["v"], # for continuous actions
            # actions_v=self.actions[0], # for discrete actions
            # beta_1=self.beta_1,
            # beta_0=self.beta_0,
            rewards=self.rewards,
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

        ##==== get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()

        ##==== calculating State
        # image as observation
        if self.state_space == "image":
            state = np.array(
                self.f1gazeboimages.image_preprocessing_black_white_32x32(
                    f1_image_camera.data, self.height
                )
            )
            state_size = state.shape

        # simplified perception as observation
        else:
            (
                centrals_in_pixels,
                centrals_normalized,
            ) = self.simplifiedperception.calculate_centrals_lane(
                f1_image_camera.data,
                self.height,
                self.width,
                self.x_row,
                self.lower_limit,
                self.center_image,
            )
            states = self.simplifiedperception.calculate_observation(
                centrals_in_pixels, self.center_image, self.pixel_region
            )
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

        ##==== get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points, centrals_normalized = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points[self.poi]
        else:
            self.point = points[0]
        # center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        center = float(self.center_image - self.point) / (float(self.width) // 2)

        ##==== get State
        ##==== image as observation
        if self.state_space == "image":
            state = np.array(
                self.f1gazeboimages.image_preprocessing_black_white_32x32(
                    f1_image_camera.data, self.height
                )
            )

        ##==== simplified perception as observation
        else:
            state = self.simplifiedperception.calculate_observation(
                points, self.center_image, self.pixel_region
            )

        ##==== get Rewards
        if self.reward_function == "follow_right_lane_center_v_step":
            reward, done = self.f1gazeborewards.rewards_followlane_v_centerline_step(
                vel_cmd, center, step, self.rewards
            )
        elif (
            self.reward_function == "follow_right_lane_center_v_w_linear"
        ):  # this reward function ONLY for continuous actions
            reward, done = self.f1gazeborewards.rewards_followlane_v_w_centerline(
                vel_cmd, center, self.rewards, self.beta_1, self.beta_0
            )
        else:
            reward, done = self.f1gazeborewards.rewards_followlane_centerline(
                center, self.rewards
            )

        return state, reward, done, {}
