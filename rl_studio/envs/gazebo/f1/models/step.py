from geometry_msgs.msg import Twist
import numpy as np

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env


class StepFollowLine(F1Env):
    def __init__(self, **config):
        self.name = config["states"]

    def step_followline_state_image_actions_discretes(self, action, step):
        print_messages("in step_followline_state_image_actions_discretes")
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points_in_red_line, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        # center = float(self.center_image - self.point) / (float(self.width) // 2)

        ##==== get State
        ##==== image as observation
        state = np.array(
            self.f1gazeboimages.image_preprocessing_black_white_32x32(
                f1_image_camera.data, self.height
            )
        )

        ##==== get Rewards
        if self.reward_function == "followline_center":
            reward, done = self.f1gazeborewards.rewards_followline_center(
                center, self.rewards
            )

        return state, reward, done, {}

    def step_followline_state_sp_actions_discretes(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points_in_red_line, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]
        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        # center = float(self.center_image - self.point) / (float(self.width) // 2)

        ##==== get State
        ##==== simplified perception as observation
        state = self.simplifiedperception.calculate_observation(
            points_in_red_line, self.center_image, self.pixel_region
        )

        ##==== get Rewards
        if self.reward_function == "followline_center":
            reward, done = self.f1gazeborewards.rewards_followlane_v_centerline_step(
                vel_cmd, center, step, self.rewards
            )

        return state, reward, done, {}


class StepFollowLane(F1Env):
    def __init__(self, **config):
        self.name = config["states"]

    def step_followlane_state_image_actions_discretes(self, action, step):
        print_messages("in step_f1_state_image_actions_discretes")
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points, _ = self.simplifiedperception.processed_image(
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
        state = np.array(
            self.f1gazeboimages.image_preprocessing_black_white_32x32(
                f1_image_camera.data, self.height
            )
        )

        ##==== get Rewards
        if self.reward_function == "follow_right_lane_center_v_step":
            reward, done = self.f1gazeborewards.rewards_followlane_v_centerline_step(
                vel_cmd, center, step, self.rewards
            )
        else:
            reward, done = self.f1gazeborewards.rewards_followlane_centerline(
                center, self.rewards
            )

        return state, reward, done, {}

    def step_followlane_state_sp_actions_discretes(self, action, step):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== get center
        points, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        if self.state_space == "spn":
            self.point = points[self.poi]
        else:
            self.point = points[0]
        # center = abs(float(self.center_image - self.point) / (float(self.width) // 2))
        center = float(self.center_image - self.point) / (float(self.width) // 2)

        ##==== get State
        ##==== simplified perception as observation
        state = self.simplifiedperception.calculate_observation(
            points, self.center_image, self.pixel_region
        )

        ##==== get Rewards
        if self.reward_function == "follow_right_lane_center_v_step":
            reward, done = self.f1gazeborewards.rewards_followlane_v_centerline_step(
                vel_cmd, center, step, self.rewards
            )
        else:
            reward, done = self.f1gazeborewards.rewards_followlane_centerline(
                center, self.rewards
            )

        return state, reward, done, {}
