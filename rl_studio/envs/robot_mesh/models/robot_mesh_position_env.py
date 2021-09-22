import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
# from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState

# from agents.robot.settings import telemetry, x_row, center_image, width, height, telemetry_mask, max_distance
from rl_studio.envs.robot_mesh.models.robot_mesh_env import MyEnv

import time
import math

class RobotMeshEnv(MyEnv):

    def __init__(self, **config):
        MyEnv.__init__(self, **config)
        print(config)
        # self.image = Imagerobot()
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(len(self.actions))  # actions  # spaces.Discrete(3)  # F,L,R

    def render(self, mode='human'):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self._gazebo_unpause()

        y_prev, x_prev, or_prev= self.get_position()

        print("pos x prev -> " + str(x_prev))
        print("pos y prev -> " + str(y_prev))

        pos_y_prev=math.ceil((y_prev+15)/5)
        pos_x_prev=math.floor((x_prev+16)/5)*7
        pos_or_prev=np.round((or_prev+4.5/1.5))

        state_prev=(pos_x_prev+pos_y_prev)

        object_coordinates = self.model_coordinates("my_robot", "")
        x_position = object_coordinates.pose.position.x
        y_position = object_coordinates.pose.position.y
        z_position = object_coordinates.pose.position.z
        x_orientation = object_coordinates.pose.orientation.x
        y_orientation = object_coordinates.pose.orientation.y
        z_orientation= object_coordinates.pose.orientation.z
        w_orientation= object_coordinates.pose.orientation.w


        state = ModelState()
        state.model_name = "my_robot"
        state.pose.position.x = x_position
        state.pose.position.y = y_position
        state.pose.position.z = z_position
        state.pose.orientation.x =self.actions[action][0]
        state.pose.orientation.y =self.actions[action][1]
        state.pose.orientation.z = self.actions[action][2]
        state.pose.orientation.w = self.actions[action][3]
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")


        vel_cmd = Twist()
        vel_cmd.linear.x = 48
        vel_cmd.angular.z = 0
        self.vel_pub.publish(vel_cmd)

        time.sleep(0.125)
        self._gazebo_pause()


        y, x, ori = self.get_position()

        print("pos x -> " + str(x))
        print("pos y -> " + str(y))

        pos_y=math.ceil((y+15)/5)
        pos_x=math.floor((x+16)/5)*7
        pos_ori=np.round((ori+4.5)/1.5)

        state=(pos_x+pos_y)
        print("¡¡¡¡¡¡¡state!!!!!!!!!! -> " + str(state))


        reward=-1
        done = False
        completed=False

        if x>self.goal:
            reward=0
            done=True
            completed=True


        return state, reward, done, completed

    def reset(self):


        self._gazebo_set_new_pose_robot()

        self._gazebo_unpause()


        y, x, ori = self.get_position()

        print("pos x -> " + str(x))
        print("pos y -> " + str(y))

        pos_y=math.ceil((y+15)/5)
        pos_x=math.floor((x+16)/5)*7
        pos_ori=np.round((ori+4.5)/1.5)

        state=(pos_x+pos_y)


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

    def finish_line(self):
        x, y, z = self.get_position()
        current_point = np.array([x, y])

        dist = (self.start_pose - current_point) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        # print(dist)
        if dist < max_distance:
            return True
        return False
