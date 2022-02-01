import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
from sensor_msgs.msg import Image
from cprint import cprint
from icecream import ic
from datetime import datetime
import time

#from agents.f1.settings import telemetry, x_row, center_image, width, height, telemetry_mask, max_distance
#from settings import x_row, center_image, width, height, telemetry_mask, max_distance
from rl_studio.envs.f1.image_f1 import ImageF1
from rl_studio.envs.f1.models.f1_env import F1Env
from rl_studio.envs.gazebo_utils import set_new_pose

ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')

class F1DDPGCameraEnv(F1Env):

    def __init__(self, **config):

        F1Env.__init__(self, **config)
        self.image = ImageF1()
        self.sensor = config['sensor']

        # Image
        self.image_resizing = config['image_resizing'] / 100
        self.height = int(config['height_image'] * self.image_resizing)
        self.width = int(config['width_image'] * self.image_resizing)
        self.center_image = int(config['center_image'] * self.image_resizing)
        self.num_regions = config['num_regions']
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config['telemetry_mask']
        self.poi = config['x_row'][0]

        # States
        self.state_space = config['state_space']
        if self.state_space == 'spn':
            self.x_row = [i for i in range(1, int(self.height/2)-1)]
            #print(f"[spn] self.x_row: {self.x_row}")
        else:
            self.x_row = config['x_row']

        # Actions
        #self.beta_1 = -(config["actions"]['w_left'] / (config["actions"]['v_max'] - config["actions"]['v_min']))
        #self.beta_0 = -(self.beta_1 * config["actions"]['v_max'])
        self.action_space = config['action_space']
        self.actions = config["actions"]
        self.beta_1 = config['beta_1']
        self.beta_0 = config['beta_0']

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config['min_reward']

        # 
        self.telemetry = config["telemetry"]

    def render(self, mode='human'):
        pass

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)


    def image_msg_to_image(self, img, cv_image):
        #ic("F1DQNCameraEnv.image_msg_to_image()")
        #print(f"\n F1QlearnCameraEnv.image_msg_to_image()\n")

        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image 

    @staticmethod
    def get_center(lines):
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            #print(f"No lines detected in the image")
            return 0

    def calculate_reward(self, error: float) -> float:

        d = np.true_divide(error, self.center_image)
        reward = np.round(np.exp(-d), 4)

        return reward


    def processed_image(self, img):
        """
        Convert img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        #line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        #_, mask = cv2.threshold(line_pre_proc, 48, 63, cv2.THRESH_BINARY) #(240 -> 255)
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY) #(240 -> 255)

        lines = [mask[self.x_row[idx], :] for idx, x in enumerate(self.x_row)]
        centrals = list(map(self.get_center, lines))

        # if centrals[-1] == 9:
        #     centrals[-1] = center_image

        if self.telemetry_mask:
            mask_points = np.zeros((self.height, self.width), dtype=np.uint8)
            for idx, point in enumerate(centrals):
                # mask_points[x_row[idx], centrals[idx]] = 255
                cv2.line(mask_points, (point, self.x_row[idx]), (point, self.x_row[idx]), (255, 255, 255), thickness=3)

            cv2.imshow("MASK", mask_points[image_middle_line:])
            cv2.waitKey(3)

        return centrals

    def calculate_observation(self, state: list) -> list:
        '''
        This is original Nacho's version. 
        I have other version. See f1_env_ddpg_camera.py
        '''
        #normalize = 40

        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.center_image - x) / self.pixel_region) + 1)
            #final_state.append(int(x / self.pixel_region) + 1)

        return final_state

    def rewards_discrete(self, center):
        if 0 <= center <= 0.2:
            reward = self.rewards['from_0_to_02']
        elif 0.2 < center <= 0.4:
            reward = self.rewards['from_02_to_04']
        else:
            reward = self.rewards['from_others']
        
        return reward

    def reward_v_w_center_linear(self, vel_cmd, center):
        '''
        Applies a linear regression between v and w
        Supposing there are a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = -B_1 * Max V
        B_1 = -(W Max / (V Max - V Min))

        w target = B_0 + B_1 * v
        error = w_actual - w_target
        reward = int(10*(1/exp(reward))) only gets first number as a reward

        parameters: linear and angular velocity
        return: reward
        '''
        num = 10
        w_target = self.beta_0 + (self.beta_1 * np.abs(vel_cmd.linear.x))
        error = np.abs(w_target - np.abs(vel_cmd.angular.z))
        reward = num * (1/np.exp(error))
        reward = reward / (center + 0.01) #Maximize near center and avoid zero in denominator
        return reward


    def reset(self):
        '''
        Main reset. Depending of:
        - sensor
        - states: images or simplified perception (sp)

        '''
        #ic(sensor)
        if self.sensor == 'camera':
            return self.reset_camera()  


    def reset_camera(self):
        if self.alternate_pose:
            print(f"\n[INFO] ===> Necesary implement self._gazebo_set_new_pose()...class F1DDPGCameraEnv(F1Env) -> def reset_camera() \n")
            #self._gazebo_set_new_pose() # Mine, it works fine!!!
            #pos_number = set_new_pose(self.circuit_positions_set) #not using. Just for V1.1
        else:
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        image_data = None
        f1_image_camera = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            # now resizing the image
            cv_image = cv2.resize(cv_image, (int(cv_image.shape[1] * self.image_resizing), int(cv_image.shape[0] * self.image_resizing)), cv2.INTER_AREA)             
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            
            #f1_image_camera = cv_image
            if np.any(cv_image):
                success = True

        # veamos que center me da en el reset()
        points = self.processed_image(f1_image_camera.data)
        self._gazebo_pause()

        if self.state_space == 'image':
            state = np.array(cv_image)
            state_size = state.shape
            return state, state_size
        else:
            state = self.calculate_observation(points)
            state_size = len(state)
            return state, state_size 


    def step(self, action):   
        self._gazebo_unpause()
        vel_cmd = Twist()

        if self.action_space == 'continuous':
            vel_cmd.linear.x = action[0][0]
            vel_cmd.angular.z = action[0][1]
        else:
            vel_cmd.linear.x = self.actions[action][0]
            vel_cmd.angular.z = self.actions[action][1]
            
        self.vel_pub.publish(vel_cmd)

        # Get camera info
        image_data = None
        f1_image_camera = None
        while image_data is None:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            # now resizing the image
            cv_image = cv2.resize(cv_image, (int(cv_image.shape[1] * self.image_resizing), int(cv_image.shape[0] * self.image_resizing)), cv2.INTER_AREA)             
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            

        self._gazebo_pause()
        points = self.processed_image(f1_image_camera.data)
        if self.state_space == 'spn':
            self.point = points[self.poi]
        else:
            self.point = points[0]    

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        #state = self.calculate_observation(points)
        #state = np.array(cv_image)
        if self.state_space == 'image':
            state = np.array(cv_image)
        else:
            #state = self.calculate_observation(points)
            state = self.calculate_observation(points)

        done = False
        # calculate reward
        if center > 0.9:
            done = True
            reward = self.rewards['from_done']
        else:
            reward = self.rewards_discrete(center)

        return state, reward, done, {}
