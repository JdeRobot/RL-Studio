import rospy
import numpy as np

from gym import spaces

from gym_gazebo.envs import gazebo_env
from gazebo_msgs.srv import SetModelState, GetModelState

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from agents.f1.settings import actions, envs_params


class F1Env(gazebo_env.GazeboEnv):

    def __init__(self):
        gazebo_env.GazeboEnv.__init__(self, self.circuit["launch"])
        self.circuit = envs_params["simple"]
        self.vel_pub = rospy.Publisher('/F1ROS/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(len(actions))  # actions  # spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.position = None
        self.start_pose = np.array(self.circuit["start_pose"])
        self._seed()

    def render(self, mode='human'):
        pass

    def step(self, action):

        raise NotImplementedError

    def reset(self):

        raise NotImplementedError

    def inference(self, action):

        raise NotImplementedError
