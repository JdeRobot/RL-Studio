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
from rl_studio.envs.carla.followlane.followlane_env import FollowLaneEnv
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig


class FollowLaneQlearn(FollowLaneEnv):
    def __init__(self, **config):

        print(f"in FollowLaneQlearn\n")   
        print(f"launching FollowLaneEnv\n ")   
        ###### init F1env
        FollowLaneEnv.__init__(self, **config)
        ###### init class variables
        print(f"leaving FollowLaneEnv\n ")   
        print(f"launching FollowLaneCarlaConfig\n ")   
        FollowLaneCarlaConfig.__init__(self, **config)

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