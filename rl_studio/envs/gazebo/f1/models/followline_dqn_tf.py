#############################################
# - Task: Follow Line
# - Algorithm: DQN
# - actions: discrete
# - State: Simplified perception and raw image
#
############################################

from geometry_msgs.msg import Twist
import numpy as np

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env
from rl_studio.envs.gazebo.f1.models.settings import F1GazeboTFConfig


class FollowLineDQNF1GazeboTF(F1Env):
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


"""
class _FollowLaneDQNF1GazeboTF(F1Env):
    def __init__(self, **config):

        ###### init F1env
        F1Env.__init__(self, **config)
        ###### init class variables
        F1GazeboTFConfig.__init__(self, **config)

        print_messages(
            "FollowLaneDQNF1GazeboTF()",
            actions=self.actions,
            len_actions=len(self.actions),
            # actions_v=self.actions["v"], # for continuous actions
            # actions_v=self.actions[0], # for discrete actions
            # beta_1=self.beta_1,
            # beta_0=self.beta_0,
            rewards=self.rewards,
        )

    #########
    # reset
    #########

    def reset(self):

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
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
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

    #########
    # step
    #########
    def step(self, action, step):
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
        else:
            reward, done = self.f1gazeborewards.rewards_followlane_centerline(
                center, self.rewards
            )

        return state, reward, done, {}

"""
