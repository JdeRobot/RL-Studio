import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
# from sensor_msgs.msg import Image
# import gazebo_msgs.msg
# from gazebo_msgs.msg import geometry_msgs
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, ApplyJointEffort
from std_msgs.msg import Float64
from gazebo_msgs.srv import SetPhysicsProperties
from std_srvs.srv import Empty


# from agents.robot.settings import telemetry, x_row, center_image, width, height, telemetry_mask, max_distance
from . import gazebo_envs

import time
import math

class MountainCarEnv(gazebo_envs.GazeboEnv):

    def __init__(self, **config):
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(len(self.actions))  # actions  # spaces.Discrete(3)  # F,L,R
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
        self.apply_joint_effort = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.position = None
        self.start_pose = np.array(config.get("start_pose"))
        self._seed()

    def render(self, mode='human'):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        lap_completed=False
        if action < 0:
            return [0, 0], 0, True, lap_completed

        state=[]
        self._gazebo_unpause()

        # vel_cmd = Twist()
        # vel_cmd.linear.x = self.actions[action]
        # vel_cmd.angular.z = 0
        # self.vel_pub.publish(vel_cmd)

        try:
            effort = self.actions[action]
            start_time = rospy.Duration.from_sec(0)
            duration = rospy.Duration.from_sec(1)
            #Effort just applied to one wheel because otherwise the car will spin due to the difference of start time (ms)
            self.apply_joint_effort("left_wheel_back", effort[0], start_time, duration)
            self.apply_joint_effort("right_wheel_back", effort[1], start_time, duration)
            self.apply_joint_effort("wheel_front_left_steer", effort[2], start_time, duration)
            self.apply_joint_effort("wheel_front_right_steer", effort[3], start_time, duration)
        except rospy.ServiceException as e:
            print("Service did not process request: {}".format(str(e)))

        time.sleep(0.2)
        # self._gazebo_pause()

        object_coordinates = self.model_coordinates("my_robot", "")
        x_position = object_coordinates.pose.position.x
        y_position = object_coordinates.pose.position.y
        z_position = object_coordinates.pose.position.z
        x_orientation = object_coordinates.pose.orientation.x
        y_orientation = object_coordinates.pose.orientation.y
        z_orientation= object_coordinates.pose.orientation.z
        w_orientation= object_coordinates.pose.orientation.w
        x_linear_vel=object_coordinates.twist.linear.x
        y_linear_vel=object_coordinates.twist.linear.y

        # y, x, ori = self.get_position()

        print("pos x -> " + str(x_position))
        print("pos y -> " + str(y_position))

        pos_x = round(x_position*2)
        vel = round(x_linear_vel*2)

        #assign state
        state.append(pos_x)
        state.append(vel)

        print("vel!!!!!!!!!! -> " + str(vel))
        print("pos_x!!!!!!!!!! -> " + str(pos_x))

        done = False
        reward=0

        #Give reward
        if pos_x>self.goal:
            done=True
            lap_completed=True
            print("Car has reached the goal")
            reward=1
        # elif z_position>=3:
        #     reward=(z_position)**2
        # if z_orientation<-0.15 or z_orientation>0.15:
        #     reward=reward*1.5
        else:
            reward=0


        # if z_orientation<-0.3 or z_orientation>0.3:
        #     done=True
        if y_position<-3.56 or y_position>-0.43:
            done=True

        return state, reward, done, lap_completed

    def reset(self):


        self._gazebo_set_new_pose_robot()

        self._gazebo_unpause()

        #
        # set_gravity = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        #
        # time_step = Float64(0.001)
        # max_update_rate = Float64(1000.0)
        # gravity = geometry_msgs.msg.Vector3()
        # gravity.x = 0.0
        # gravity.y = 0.0
        # gravity.z = 0.0
        # ode_config = gazebo_msgs.msg.ODEPhysics()
        # ode_config.auto_disable_bodies = False
        # ode_config.sor_pgs_precon_iters = 0
        # ode_config.sor_pgs_iters = 50
        # ode_config.sor_pgs_w = 1.3
        # ode_config.sor_pgs_rms_error_tol = 0.0
        # ode_config.contact_surface_layer = 0.001
        # ode_config.contact_max_correcting_vel = 0.0
        # ode_config.cfm = 0.0
        # ode_config.erp = 0.2
        # ode_config.max_contacts = 20
        # set_gravity(time_step.data, max_update_rate.data, gravity, ode_config)

        # x, y, ori = self.get_position()

        object_coordinates = self.model_coordinates("my_robot", "")
        x_position = object_coordinates.pose.position.x
        y_position = object_coordinates.pose.position.y
        z_position = object_coordinates.pose.position.z
        x_orientation = object_coordinates.pose.orientation.x
        y_orientation = object_coordinates.pose.orientation.y
        z_orientation= object_coordinates.pose.orientation.z
        w_orientation= object_coordinates.pose.orientation.w
        x_linear_vel=object_coordinates.twist.linear.x
        y_linear_vel=object_coordinates.twist.linear.y

        print("pos x -> " + str(x_position))
        print("pos y -> " + str(y_position))
        pos_x = round(x_position)
        vel = round(x_linear_vel)

        state=[]

        state.append(pos_x)
        state.append(vel)
        #Note that state[2]=3 means that orientation is between 1.5 and -1.5 which must be true in start pos
        state.append(3)

        done=False

        # self._gazebo_pause()

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
