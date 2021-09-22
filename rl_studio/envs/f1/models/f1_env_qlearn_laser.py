import rospy
import time
import random
import numpy as np

from gym import spaces
from gym.utils import seeding

from gym_gazebo.envs import gazebo_env
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan

from gym_gazebo.agents.f1.settings import actions
from gym_gazebo.agents.f1.settings import envs_params


class F1QlearnLaserEnv(gazebo_env.GazeboEnv):
    def __init__(self, **config):
        # Launch the simulation with the given launchfile name
        self.circuit = envs_params["simple"]
        gazebo_env.GazeboEnv.__init__(self, self.circuit["launch"])
        # gazebo_env.GazeboEnv.__init__(self, "f1_1_nurburgrinlineROS_laser.launch")
        self.vel_pub = rospy.Publisher("/F1ROS/cmd_vel", Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.action_space = spaces.Discrete(
            3
        )  # actions  # spaces.Discrete(5)  # F, L, R
        self.reward_range = (-np.inf, np.inf)
        self.position = None
        self._seed()

    def render(self, mode="human"):
        pass

    def get_position(self):
        object_coordinates = self.model_coordinates("f1_renault", "")
        x_position = round(object_coordinates.pose.position.x, 2)
        y_position = round(object_coordinates.pose.position.y, 2)

        return x_position, y_position

    def _gazebo_pause(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed: {}".format(e))

    def _gazebo_unpause(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print(e)
            print("/gazebo/unpause_physics service call failed")

    def _gazebo_reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            # reset_proxy.call()
            self.reset_proxy()
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed: {}".format(e))

    def set_new_pose(self):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        pos = random.choice(list(enumerate(self.circuit["gaz_pos"])))[0]
        self.position = pos

        pos_number = self.circuit["gaz_pos"][0]

        state = ModelState()
        state.model_name = "f1_renault"
        state.pose.position.x = self.circuit["gaz_pos"][pos][1]
        state.pose.position.y = self.circuit["gaz_pos"][pos][2]
        state.pose.position.z = self.circuit["gaz_pos"][pos][3]
        state.pose.orientation.x = self.circuit["gaz_pos"][pos][4]
        state.pose.orientation.y = self.circuit["gaz_pos"][pos][5]
        state.pose.orientation.z = self.circuit["gaz_pos"][pos][6]
        state.pose.orientation.w = self.circuit["gaz_pos"][pos][7]

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))
        return pos_number

    @staticmethod
    def discrete_observation(data, new_ranges):

        discrete_ranges = []
        min_range = 0.05
        done = False
        mod = len(data.ranges) / new_ranges
        filter_data = data.ranges[10:-10]
        for i, item in enumerate(filter_data):
            if i % mod == 0:
                if filter_data[i] == float("Inf") or np.isinf(filter_data[i]):
                    discrete_ranges.append(6)
                elif np.isnan(filter_data[i]):
                    discrete_ranges.append(0)
                else:
                    discrete_ranges.append(int(filter_data[i]))
            if min_range > filter_data[i] > 0:
                print("Data ranges: {}".format(data.ranges[i]))
                done = True
                break

        return discrete_ranges, done

    @staticmethod
    def get_center_of_laser(data):

        laser_len = len(data.ranges)
        left_sum = sum(
            data.ranges[laser_len - (laser_len / 5) : laser_len - (laser_len / 10)]
        )  # 80-90
        right_sum = sum(data.ranges[(laser_len / 10) : (laser_len / 5)])  # 10-20

        center_detour = (right_sum - left_sum) / 5

        return center_detour

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self._gazebo_unpause()

        vel_cmd = Twist()
        vel_cmd.linear.x = actions[action][0]
        vel_cmd.angular.z = actions[action][1]
        self.vel_pub.publish(vel_cmd)

        laser_data = None
        success = False
        while laser_data is None or not success:
            try:
                laser_data = rospy.wait_for_message(
                    "/F1ROS/laser/scan", LaserScan, timeout=5
                )
            finally:
                success = True

        self._gazebo_pause()

        state, _ = self.discrete_observation(laser_data, 5)

        laser_len = len(laser_data.ranges)
        left_sum = sum(
            laser_data.ranges[
                laser_len - (laser_len / 5) : laser_len - (laser_len / 10)
            ]
        )  # 80-90
        right_sum = sum(laser_data.ranges[(laser_len / 10) : (laser_len / 5)])  # 10-20
        left_boundary = left_sum / 5
        right_boundary = right_sum / 5
        center_detour = right_boundary - left_boundary

        # print("LEFT: {} - RIGHT: {}".format(left_boundary, right_boundary))
        # print(state)

        done = False
        if abs(center_detour) > 2 or left_boundary < 2 or right_boundary < 2:
            done = True
        # print("center: {} - Action: {}".format(center_detour, action))

        if not done:
            if 0.2 < abs(center_detour) < 0.4:
                reward = 5
            elif abs(center_detour) < 0.2 and action == 0:
                reward = 10
            else:  # L or R no looping
                reward = 2
        else:
            reward = -200

        print(
            "center: {} - actions: {} - reward: {}".format(
                center_detour, action, reward
            )
        )

        return state, reward, done, {}

    def reset(self):
        # === POSE ===
        # self.set_new_pose()
        self._gazebo_reset()
        time.sleep(0.1)

        self._gazebo_unpause()

        # Read laser data
        laser_data = None
        success = False
        while laser_data is None or not success:
            try:
                laser_data = rospy.wait_for_message(
                    "/F1ROS/laser/scan", LaserScan, timeout=5
                )
            finally:
                success = True

        self._gazebo_pause()

        state = self.discrete_observation(laser_data, 5)

        return state
