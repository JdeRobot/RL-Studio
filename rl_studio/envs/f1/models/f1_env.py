import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Twist
from gym import spaces
from std_srvs.srv import Empty

from rl_studio.envs import gazebo_env


class F1Env(gazebo_env.GazeboEnv):
    def __init__(self, **config):
        gazebo_env.GazeboEnv.__init__(self, config.get("launch"))
        self.circuit = config.get("simple")
        self.alternate_pose = config.get("algernate_pose")
        self.vel_pub = rospy.Publisher("/F1ROS/cmd_vel", Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        self.position = None
        self.start_pose = np.array(config.get("start_pose"))
        self._seed()

    def render(self, mode="human"):
        pass

    def step(self, action):

        raise NotImplementedError

    def reset(self):

        raise NotImplementedError

    def inference(self, action):

        raise NotImplementedError
