import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Twist
from gym import spaces
from std_srvs.srv import Empty

from gym_gazebo.envs import gazebo_env
from gym_gazebo.envs.f1.env_type import TrainingType
from gym_gazebo.envs.f1.exceptions import NoValidTrainingType


from cprint import cprint


class F1Env(gazebo_env.GazeboEnv):

    def __init__(self, **config):

        cprint.warn(f"\n [F1Env] -> --------- Enter in F1Env ------- \n")

        # Launch GazeboEnv
        #gazebo_env.GazeboEnv.__init__(self, config.get("launch"))
        gazebo_env.GazeboEnv.__init__(self, **config)


        #self.circuit = config.get("simple")
        #print(f"\n[F1Env] -> self.circuit = {self.circuit}\n")

        #print(f"-------[F1Env] error in next line algernate_pose ---------")
        self.alternate_pose = config.get("alternate_pose")
        #print(f"\n[F1Env] -> self.alternate_pose = {self.alternate_pose}\n")


        # --------------------Launch TOPICS
        self.vel_pub = rospy.Publisher('/F1ROS/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.position = None
        self.start_pose = np.array(config.get("start_pose"))
        self._seed()

        cprint.ok(f"\n  [F1Env] -> -------Out F1Env (__init__) ------------\n")


    def render(self, mode='human'):
        pass

    def step(self, action):

        raise NotImplementedError

    def reset(self):

        raise NotImplementedError

    def inference(self, action):

        raise NotImplementedError
