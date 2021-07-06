import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
from sensor_msgs.msg import Image

#from agents.f1.settings import telemetry, x_row, center_image, width, height, telemetry_mask, max_distance
from settings import telemetry, x_row, center_image, width, height, telemetry_mask, max_distance
from gym_gazebo.envs.f1.image_f1 import ImageF1
from gym_gazebo.envs.f1.models.f1_env import F1Env

from cprint import cprint
from icecream import ic
from datetime import datetime


ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')

class F1DQNCameraEnv(F1Env):

    def __init__(self, **config):

        #cprint.warn(f"\n [F1DQNCameraEnv] -> --------- Enter in F1DQNCameraEnv ---------------\n")
        ic('Enter in F1DQNCameraEnv')
        F1Env.__init__(self, **config)
        #print(f"\n [F1DQNCameraEnv] -> config: {config}")
        self.image = ImageF1()
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(len(self.actions))  # actions  # spaces.Discrete(3)  # F,L,R

        self.rewards = config["rewards"]
        #ic(self.rewards)
        #ic(self.rewards['from_done'])


        cprint.ok(f"\n  [F1DQNCameraEnv] -> ------------ Out F1DQNCameraEnv (__init__) -----------\n")


    def reset(self):
        #print(f"\n F1QlearnCameraEnv.reset()\n")
        ic("F1DQNCameraEnv.reset()")
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_new_pose
            #self.set_new_pose()
        else:
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        image_data = None
        f1_image_camera = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            #f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            #f1_image_camera = cv_image
            #if f1_image_camera:
            #    success = True

        #points = self.processed_image(f1_image_camera.data)
        #state = self.calculate_observation(points)
        # reset_state = (state, False)

        self._gazebo_pause()

        #return state
        return np.array(cv_image)


    def image_msg_to_image(self, img, cv_image):
        #print(f"\n F1QlearnCameraEnv.image_msg_to_image()\n")

        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image        