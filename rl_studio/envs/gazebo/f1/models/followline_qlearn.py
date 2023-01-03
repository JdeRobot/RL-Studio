#############################################
# - Task: Follow Line
# - Algorithm: Qlearn
# - actions: discrete
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
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env
from rl_studio.envs.gazebo.f1.models.settings import F1GazeboTFConfig


class FollowLineQlearnF1Gazebo(F1Env):
    def __init__(self, **config):

        ###### init F1env
        F1Env.__init__(self, **config)
        ###### init class variables
        F1GazeboTFConfig.__init__(self, **config)

    def reset(self):
        from rl_studio.envs.gazebo.f1.models.reset import (
            Reset,
        )

        if self.state_space == "image":
            return Reset.reset_f1_state_image(self)
        else:
            return Reset.reset_f1_state_sp(self)

    def step(self, action, step):
        from rl_studio.envs.gazebo.f1.models.step import (
            StepFollowLine,
        )

        if self.state_space == "image":
            return StepFollowLine.step_followline_state_image_actions_discretes(
                self, action, step
            )
        else:
            return StepFollowLine.step_followline_state_sp_actions_discretes(
                self, action, step
            )


'''
class FollowLineQlearnF1Gazebo(F1Env):
    def __init__(self, **config):

        F1Env.__init__(self, **config)
        self.simplifiedperception = F1GazeboSimplifiedPerception()
        self.f1gazeborewards = F1GazeboRewards()
        self.f1gazeboutils = F1GazeboUtils()
        self.f1gazeboimages = F1GazeboImages()

        self.image = ImageF1()
        self.f1_image_camera = None
        self.sensor = config["sensor"]

        # Image
        self.height = config["height_image"]
        self.width = config["width_image"]
        self.center_image = config["center_image"]
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config["telemetry_mask"]
        self.telemetry = config["telemetry"]
        self.poi = config["x_row"][0]
        self.image_center = None

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

        print_messages(
            "FollowLineQlearnF1Gazebo()",
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
        if self.sensor == "camera":
            return self.reset_camera()

    def reset_camera(self):
        self._gazebo_reset()
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_followline()
        else:
            self._gazebo_set_fix_pose_f1_followline()

        self._gazebo_unpause()

        ##==== get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()

        ##==== calculating State
        # simplified perception as observation
        points, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        state = self.simplifiedperception.calculate_observation(
            points, self.center_image, self.pixel_region
        )
        # state = [states[0]]
        state_size = len(state)

        print_messages(
            "reset()",
            state=state,
            state_size=state_size,
        )

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

    def step(self, action):

        self._gazebo_unpause()

        vel_cmd = Twist()
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
        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        ##==== get State
        ##==== simplified perception as observation
        state = self.simplifiedperception.calculate_observation(
            points, self.center_image, self.pixel_region
        )

        ##==== get Rewards
        if self.reward_function == "followline_center":
            reward, done = self.f1gazeborewards.rewards_followline_center(
                center, self.rewards
            )
        else:
            reward, done = self.f1gazeborewards.rewards_followline_v_w_centerline(
                vel_cmd, center, self.rewards, self.beta_1, self.beta_0
            )
        return state, reward, done, {}
'''
