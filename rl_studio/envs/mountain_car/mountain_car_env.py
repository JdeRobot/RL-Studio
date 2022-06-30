import time

import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState, ApplyJointEffort
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
from std_srvs.srv import Empty

from .. import gazebo_envs


class MountainCarEnv(gazebo_envs.GazeboEnv):
    def __init__(self, **config):
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(
            len(self.actions)
        )  # actions  # spaces.Discrete(3)  # F,L,R
        gazebo_envs.GazeboEnv.__init__(self, config)
        self.circuit = config.get("simple")
        self.alternate_pose = config.get("alternate_pose")
        self.goal = config.get("goal")
        self.reset_pos_x = config.get("pos_x")
        self.reset_pos_y = config.get("pos_y")
        self.reset_pos_z = config.get("pos_z")
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.apply_joint_effort = rospy.ServiceProxy(
            "/gazebo/apply_joint_effort", ApplyJointEffort
        )
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        self.position = None
        self.start_pose = np.array(config.get("start_pose"))
        self._seed()

    def render(self, mode="human"):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        lap_completed = False
        if action < 0:
            return [0, 0], 0, True, lap_completed

        state = []
        self._gazebo_unpause()


        try:
            effort = self.actions[action]
            start_time = rospy.Duration.from_sec(0)
            duration = rospy.Duration.from_sec(1)
            # Effort just applied to one wheel because otherwise the car will spin due to the difference of start time (ms)
            self.apply_joint_effort("left_wheel_back", effort[0], start_time, duration)
            self.apply_joint_effort("right_wheel_back", effort[1], start_time, duration)
            self.apply_joint_effort(
                "wheel_front_left_steer", effort[2], start_time, duration
            )
            self.apply_joint_effort(
                "wheel_front_right_steer", effort[3], start_time, duration
            )
        except rospy.ServiceException as e:
            print("Service did not process request: {}".format(str(e)))

        time.sleep(0.2)
        # self._gazebo_pause()

        object_coordinates = self.model_coordinates("my_robot", "")
        x_position = object_coordinates.pose.position.x
        y_position = object_coordinates.pose.position.y

        x_linear_vel = object_coordinates.twist.linear.x


        print("pos x -> " + str(x_position))
        print("pos y -> " + str(y_position))

        pos_x = round(x_position * 2)
        vel = round(x_linear_vel * 2)

        # assign state
        state.append(pos_x)
        state.append(vel)

        print("vel!!!!!!!!!! -> " + str(vel))
        print("pos_x!!!!!!!!!! -> " + str(pos_x))

        done = False
        reward = 0

        # Give reward
        if pos_x > self.goal:
            done = True
            lap_completed = True
            print("Car has reached the goal")
            reward = 1

        else:
            reward = 0


        if y_position < -3.56 or y_position > -0.43:
            done = True

        return state, reward, done, lap_completed

    def reset(self):

        self._gazebo_set_new_pose_robot()

        self._gazebo_unpause()


        object_coordinates = self.model_coordinates("my_robot", "")
        x_position = object_coordinates.pose.position.x
        y_position = object_coordinates.pose.position.y

        x_linear_vel = object_coordinates.twist.linear.x

        print("pos x -> " + str(x_position))
        print("pos y -> " + str(y_position))
        pos_x = round(x_position)
        vel = round(x_linear_vel)

        state = []

        state.append(pos_x)
        state.append(vel)
        # Note that state[2]=3 means that orientation is between 1.5 and -1.5 which must be true in start pos
        state.append(3)

        done = False


        return state

    def finish_line(self):
        x, y = self.get_position()
        current_point = np.array([x, y])

        dist = (self.start_pose - current_point) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)

        if dist < max_distance:
            return True
        return False
