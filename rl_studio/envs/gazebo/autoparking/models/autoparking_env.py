import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

from rl_studio.envs.gazebo import gazebo_envs


class AutoparkingEnv(gazebo_envs.GazeboEnv):
    def __init__(self, **config):
        gazebo_envs.GazeboEnv.__init__(self, config)
        self.circuit_name = config.get("circuit_name")
        # self.circuit_positions_set = config.get("circuit_positions_set")
        self.alternate_pose = config.get("alternate_pose")
        self.model_state_name = config.get("model_state_name")
        self.position = None
        # self.start_pose = np.array(config.get("alternate_pose"))
        self.start_pose = np.array(config.get("gazebo_start_pose"))
        self.start_random_pose = config.get("gazebo_random_start_pose")
        self._seed()
        self.parking_spot_position_x = config.get("parking_spot_position_x")
        self.parking_spot_position_y = config.get("parking_spot_position_y")

        # self.cv_image_pub = rospy.Publisher('/F1ROS/cameraL/image_raw', Image, queue_size = 10)
        self.vel_pub = rospy.Publisher("/F1ROS/cmd_vel", Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
