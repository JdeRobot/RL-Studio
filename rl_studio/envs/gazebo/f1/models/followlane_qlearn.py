#############################################
# - Task: Follow Lane
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


class FollowLaneQlearnF1Gazebo(F1Env):
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
            StepFollowLane,
        )

        return StepFollowLane.step_followlane_state_sp_actions_discretes(
            self, action, step
        )
