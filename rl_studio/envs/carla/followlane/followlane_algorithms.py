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
        from rl_studio.envs.carla.followlane.followlane_env import (
            FollowLaneEnv,
        )
        return FollowLaneEnv.reset(self)

    def step(self, action, step):
        from rl_studio.envs.carla.followlane.followlane_env import (
            FollowLaneEnv,
        )
        return FollowLaneEnv.step(
            self, action, step
        )