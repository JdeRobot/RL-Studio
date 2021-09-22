import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Twist
from gym import spaces
from std_srvs.srv import Empty

from rl_studio.envs.robot_mesh import gazebo_envs

class MyEnv(gazebo_envs.GazeboEnv):

    def __init__(self, **config):
        gazebo_envs.GazeboEnv.__init__(self, config.get("launch"))
        self.circuit = config.get("simple")
        self.alternate_pose = config.get("alternate_pose")
        self.goal=config.get("goal")
        self.reset_pos_x=config.get("pos_x")
        self.reset_pos_y=config.get("pos_y")
        self.reset_pos_z=config.get("pos_z")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.position = None
        self.start_pose = np.array(config.get("start_pose"))
        self._seed()

    def render(self, mode='human'):
        pass

    def step(self, action):

        raise NotImplementedError

    def reset(self):

        raise NotImplementedError

    def inference(self, action):

        raise NotImplementedError
