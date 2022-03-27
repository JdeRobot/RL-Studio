import numpy as np
import rospy
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding

# from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from std_srvs.srv import Empty

# from agents.robot.settings import telemetry, x_row, center_image, width, height, telemetry_mask, max_distance
from .. import gazebo_envs
import time
import math


def euclidean_distance(x_a, x_b, y_a, y_b):
    return np.sqrt((x_b - x_a) ** 2 + (y_b - y_a) ** 2)


class RobotMeshEnv(gazebo_envs.GazeboEnv):
    def __init__(self, **config):
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(
            len(self.actions)
        )  # actions  # spaces.Discrete(3)  # F,L,R
        gazebo_envs.GazeboEnv.__init__(self, config)
        self.circuit = config.get("simple")
        self.alternate_pose = config.get("alternate_pose")
        self.actions_force = config.get("actions_force")
        self.boot_on_crash = config.get("boot_on_crash")
        self.goal_x = config.get("goal_x")
        self.goal_y = config.get("goal_y")
        self.robot_name = config.get("robot_name")
        self.reset_pos_x = config.get("pos_x")
        self.reset_pos_y = config.get("pos_y")
        self.reset_pos_z = config.get("pos_z")
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
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
        self.movement_precision = 0.6
        self.cells_span = self.actions_force / 10

    def render(self, mode="human"):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        x_prev, y_prev = self.get_position()

        print("pos x prev -> " + str(x_prev))
        print("pos y prev -> " + str(y_prev))

        pos_x_prev = math.ceil((x_prev + 10) / self.cells_span)
        pos_y_prev = math.ceil((y_prev + 10) / self.cells_span) * 40
        # pos_or_prev=np.round((or_prev+4.5/self.cells_span))

        object_coordinates = self.model_coordinates(self.robot_name, "")
        x_position = object_coordinates.pose.position.x
        y_position = object_coordinates.pose.position.y
        z_position = object_coordinates.pose.position.z
        x_orientation = object_coordinates.pose.orientation.x
        y_orientation = object_coordinates.pose.orientation.y
        z_orientation = object_coordinates.pose.orientation.z
        w_orientation = object_coordinates.pose.orientation.w

        state = ModelState()
        state.model_name = self.robot_name
        state.pose.position.x = x_position
        state.pose.position.y = y_position
        state.pose.position.z = z_position
        state.pose.orientation.x = self.actions[action][0]
        state.pose.orientation.y = self.actions[action][1]
        state.pose.orientation.z = self.actions[action][2]
        state.pose.orientation.w = self.actions[action][3]
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions_force
        vel_cmd.angular.z = 0

        self.vel_pub.publish(vel_cmd)

        self._gazebo_unpause()

        time.sleep(0.125)

        self._gazebo_pause()

        x, y = self.get_position()

        print("pos x -> " + str(x))
        print("pos y -> " + str(y))

        pos_x = math.ceil((x + 10) / self.cells_span)
        pos_y = math.ceil((y + 10) / self.cells_span) * 40
        # pos_ori=np.round((ori+4.5)/self.cells_span)

        state = pos_x + pos_y
        print("¡¡¡¡¡¡¡state!!!!!!!!!! -> " + str(state))

        reward = -1
        done = False
        completed = False

        if (
            euclidean_distance(x_prev, x, y_prev, y)
            < self.movement_precision * self.cells_span
            and self.boot_on_crash
        ):
            reward = -1
            done = True
            completed = False

        if x < self.goal_x or y > self.goal_y:
            reward = 0
            done = True
            completed = True

        return state, reward, done, completed

    def reset(self):

        self._gazebo_set_new_pose_robot()

        self._gazebo_unpause()

        y, x = self.get_position()

        print("pos x -> " + str(x))
        print("pos y -> " + str(y))

        pos_x = math.ceil((x + 10) / self.cells_span)
        pos_y = math.ceil((y + 10) / self.cells_span) * 40
        # pos_ori=np.round((ori+4.5)/self.cells_span)

        state = pos_x + pos_y

        self._gazebo_pause()

        return state

    # def inference(self, action):
    #     self._gazebo_unpause()
    #
    #     vel_cmd = Twist()
    #     vel_cmd.linear.x = ACTIONS_SET[action][0]
    #     vel_cmd.angular.z = ACTIONS_SET[action][1]
    #     self.vel_pub.publish(vel_cmd)
    #
    #     image_data = rospy.wait_for_message('/robotROS/cameraL/image_raw', Image, timeout=1)
    #     cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
    #     robot_image_camera = self.image_msg_to_image(image_data, cv_image)
    #
    #     self._gazebo_pause()
    #
    #     points = self.processed_image(robot_image_camera.data)
    #     state = self.calculate_observation(points)
    #
    #     center = float(center_image - points[0]) / (float(width) // 2)
    #
    #     done = False
    #     center = abs(center)
    #
    #     if center > 0.9:
    #         done = True
    #
    #     return state, done
