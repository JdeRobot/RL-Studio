import math
import random
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from gym.utils import seeding
from PIL import Image as im
import rospy
from sensor_msgs.msg import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from std_srvs.srv import Empty

from rl_studio.agents.f1.settings import qlearn
from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.gazebo.f1.image_f1 import ImageF1
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env

# Images size
witdh = 640
center_image = int(witdh / 2)

# Coord X ROW
x_row = [260, 360, 450]
# Maximum distance from the line
RANGES = [300, 280, 250]  # Line 1, 2 and 3

RESET_RANGE = [-40, 40]

# Deprecated?
space_reward = np.flip(np.linspace(0, 1, 300))

last_center_line = 0

font = cv2.FONT_HERSHEY_COMPLEX

# OUTPUTS
v_lineal = [3, 8, 15]
w_angular = [-1, -0.6, 0, 1, 0.6]

# POSES
positions = [
    (0, 53.462, -41.988, 0.004, 0, 0, 1.57, -1.57),
    (1, 53.462, -8.734, 0.004, 0, 0, 1.57, -1.57),
    (2, 39.712, -30.741, 0.004, 0, 0, 1.56, 1.56),
    (3, -7.894, -39.051, 0.004, 0, 0.01, -2.021, 2.021),
    (4, 20.043, 37.130, 0.003, 0, 0.103, -1.4383, -1.4383),
]


class GazeboF1CameraEnvDQN(F1Env):
    """
    Description:
        A Formula 1 car has to complete one lap of a circuit following a red line painted on the ground. Initially it
        will not use the complete information of the image but some coordinates that refer to the error made with
        respect to the center of the line.
    Source:
        Master's final project at Universidad Rey Juan Carlos. RoboticsLab Urjc. JdeRobot. Author: Ignacio Arranz
    Observation:
        Type: Array
        Num	Observation               Min   Max
        ----------------------------------------
        0	Vel. Lineal (m/s)         1     10
        1	Vel. Angular (rad/seg)   -2     2
        2	Error 1                  -300   300
        3	Error 2                  -280   280
        4   Error 3                  -250   250
    Actions:
        Type: Dict
        Num	Action
        ----------
        0:   -2
        1:   -1
        2:    0
        3:    1
        4:    2
    Reward:
        The reward depends on the set of the 3 errors. As long as the lowest point is within range, there will still
        be steps. If errors 1 and 2 fall outside the range, a joint but weighted reward will be posted, always giving
        more importance to the lower one.
    Starting State:
        The observations will start from different points in order to prevent you from performing the same actions
        initially. This variability will enrich the set of state/actions.
    Episode Termination:
        The episode ends when the lower error is outside the established range (see table in the observation space).
    """

    def __init__(self, **config):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "F1Cameracircuit_v0.launch")
        self.vel_pub = rospy.Publisher("/F1ROS/cmd_vel", Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        # self.state_msg = ModelState()
        # self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.position = None
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.last50actions = [0] * 50
        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 1
        # self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/F1ROS/cameraL/image_raw", Image, self.callback)
        self.action_space = self._generate_simple_action_space()

    def render(self, mode="human"):
        pass

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

    @staticmethod
    def _generate_simple_action_space():
        actions = 5
        max_ang_speed = -4
        action_space = {}

        for action in range(actions):
            if action > actions / 2:
                diff = action - round(actions / 2)
            vel_ang = round(
                (action - actions / 2) * max_ang_speed * 0.1, 2
            )  # from (-1 to + 1)
            action_space[action] = vel_ang

        return action_space

    @staticmethod
    def _generate_action_space():
        actions = 21

        max_ang_speed = 1.5
        min_lin_speed = 2
        max_lin_speed = 12

        action_space_dict = {}

        for action in range(actions):
            if action > actions / 2:
                diff = action - round(actions / 2)
                vel_lin = max_lin_speed - diff  # from (3 to 15)
            else:
                vel_lin = action + min_lin_speed  # from (3 to 15)
            vel_ang = round(
                (action - actions / 2) * max_ang_speed * 0.1, 2
            )  # from (-1 to + 1)
            action_space_dict[action] = (vel_lin, vel_ang)
            # print("Action: {} - V: {} - W: {}".format(action, vel_lin, vel_ang))
        # print(action_space_dict)

        return action_space_dict

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def show_telemetry(self, img, point_1, point_2, point_3, action, reward):
        # Puntos centrales de la imagen (verde)
        cv2.line(img, (320, x_row[0]), (320, x_row[0]), (255, 255, 0), thickness=5)
        cv2.line(img, (320, x_row[1]), (320, x_row[1]), (255, 255, 0), thickness=5)
        cv2.line(img, (320, x_row[2]), (320, x_row[2]), (255, 255, 0), thickness=5)
        # Linea diferencia entre punto central - error (blanco)
        cv2.line(
            img,
            (center_image, x_row[0]),
            (int(point_1), x_row[0]),
            (255, 255, 255),
            thickness=2,
        )
        cv2.line(
            img,
            (center_image, x_row[1]),
            (int(point_2), x_row[1]),
            (255, 255, 255),
            thickness=2,
        )
        cv2.line(
            img,
            (center_image, x_row[2]),
            (int(point_3), x_row[2]),
            (255, 255, 255),
            thickness=2,
        )
        # Telemetry
        cv2.putText(
            img,
            str("action: {}".format(action)),
            (18, 280),
            font,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            str("w ang: {}".format(w_angular)),
            (18, 300),
            font,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            str("reward: {}".format(reward)),
            (18, 320),
            font,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            str("err1: {}".format(center_image - point_1)),
            (18, 340),
            font,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            str("err2: {}".format(center_image - point_2)),
            (18, 360),
            font,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            str("err3: {}".format(center_image - point_3)),
            (18, 380),
            font,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            str("pose: {}".format(self.position)),
            (18, 400),
            font,
            0.4,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Image window", img)
        cv2.waitKey(3)

    @staticmethod
    def set_new_pose(new_pos):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        pos_number = positions[0]

        state = ModelState()
        state.model_name = "f1_renault"
        state.pose.position.x = positions[new_pos][1]
        state.pose.position.y = positions[new_pos][2]
        state.pose.position.z = positions[new_pos][3]
        state.pose.orientation.x = positions[new_pos][4]
        state.pose.orientation.y = positions[new_pos][5]
        state.pose.orientation.z = positions[new_pos][6]
        state.pose.orientation.w = positions[new_pos][7]

        rospy.wait_for_service("/gazebo/set_model_state")

        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))

        return pos_number

    @staticmethod
    def image_msg_to_image(img, cv_image):

        image = ImageF1()
        image.width = img.width
        image.height = img.height
        image.format = "RGB8"
        image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        image.data = cv_image

        return image

    @staticmethod
    def get_center(image_line):
        try:
            coords = np.divide(
                np.max(np.nonzero(image_line)) - np.min(np.nonzero(image_line)), 2
            )
            coords = np.min(np.nonzero(image_line)) + coords
        except:
            coords = -1

        return coords

    def processed_image(self, img):
        """
        Conver img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """

        img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 200))

        # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        line_1 = mask[x_row[0], :]
        line_2 = mask[x_row[1], :]
        line_3 = mask[x_row[2], :]

        central_1 = self.get_center(line_1)
        central_2 = self.get_center(line_2)
        central_3 = self.get_center(line_3)

        # print(central_1, central_2, central_3)

        return central_1, central_2, central_3

    def callback(self, data):

        # print("CALLBACK!!!!: ", ros_data.height, ros_data.width)
        # np_arr = np.fromstring(ros_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # self.my_image = image_np
        # rospy.loginfo(rospy.get_caller_id() + "I see %s", data.data)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        if cols > 60 and rows > 60:
            cv2.circle(cv_image, (50, 50), 10, 255)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)

        # try:
        #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # except CvBridgeError as e:
        #   print(e)

    @staticmethod
    def calculate_error(point_1, point_2, point_3):

        error_1 = abs(center_image - point_1)
        error_2 = abs(center_image - point_2)
        error_3 = abs(center_image - point_3)

        return error_1, error_2, error_3

    @staticmethod
    def calculate_reward(error_1, error_2, error_3):

        global center_image
        alpha = 0
        beta = 0
        gamma = 1

        # if error_1 > RANGES[0] and error_2 > RANGES[1]:
        #     ALPHA = 0.1
        #     BETA = 0.2
        #     GAMMA = 0.7
        # elif error_1 > RANGES[0]:
        #     ALPHA = 0.1
        #     BETA = 0
        #     GAMMA = 0.9
        # elif error_2 > RANGES[1]:
        #     ALPHA = 0
        #     BETA = 0.1
        #     GAMMA = 0.9

        # d = ALPHA * np.true_divide(error_1, center_image) + \
        # beta * np.true_divide(error_2, center_image) + \
        # gamma * np.true_divide(error_3, center_image)
        d = np.true_divide(error_3, center_image)
        reward = np.round(np.exp(-d), 4)

        return reward

    @staticmethod
    def is_game_over(point_1, point_2, point_3):

        done = False

        if center_image - RANGES[2] < point_3 < center_image + RANGES[2]:
            if (
                center_image - RANGES[0] < point_1 < center_image + RANGES[0]
                or center_image - RANGES[1] < point_2 < center_image + RANGES[1]
            ):
                pass  # In Line
        else:
            done = True

        return done

    def step(self, action):

        self._gazebo_unpause()

        # === ACTIONS === - 5 actions
        vel_cmd = Twist()
        vel_cmd.linear.x = 3  # self.action_space[action][0]
        vel_cmd.angular.z = self.action_space[action]  # [1]
        self.vel_pub.publish(vel_cmd)
        # print("Action: {} - V_Lineal: {} - W_Angular: {}".format(action, vel_cmd.linear.x, vel_cmd.angular.z))

        # === IMAGE ===
        image_data = None
        success = False
        cv_image = None
        f1_image_camera = None
        while image_data is None or success is False:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)

            if f1_image_camera:
                success = True

        point_1, point_2, point_3 = self.processed_image(f1_image_camera.data)

        # DONE
        done = self.is_game_over(point_1, point_2, point_3)

        self._gazebo_pause()

        self.last50actions.pop(0)  # remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        # action_sum = sum(self.last50actions)

        # == CALCULATE ERROR ==
        error_1, error_2, error_3 = self.calculate_error(point_1, point_2, point_3)

        # == REWARD ==
        if not done:
            reward = self.calculate_reward(error_1, error_2, error_3)
        else:
            reward = -200

        # == TELEMETRY ==
        if qlearn.telemetry:
            self.show_telemetry(
                f1_image_camera.data, point_1, point_2, point_3, action, reward
            )

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        observation = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])

        # info = [vel_cmd.linear.x, vel_cmd.angular.z, error_1, error_2, error_3]
        # OpenAI standard return: observation, reward, done, info
        return observation, reward, done, {}

        # test STACK 4
        # cv_image = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        # self.s_t = np.append(cv_image, self.s_t[:, :3, :, :], axis=1)
        # return self.s_t, reward, done, {} # observation, reward, done, info

    def reset(self):

        self.last50actions = [0] * 50  # used for looping avoidance

        # === POSE ===
        pos = random.choice(list(enumerate(positions)))[0]
        self.position = pos
        self.set_new_pose(pos)

        # === RESET ===
        # Resets the state of the environment and returns an initial observation.
        time.sleep(0.05)
        # self._gazebo_reset()
        self._gazebo_unpause()

        image_data = None
        cv_image = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            if f1_image_camera:
                success = True

        self._gazebo_pause()

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        # cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        # cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state, pos

        # test STACK 4
        # self.s_t = np.stack((cv_image, cv_image, cv_image, cv_image), axis=0)
        # self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])
        # return self.s_t


class DQNF1FollowLineEnvGazebo(F1Env):
    def __init__(self, **config):

        F1Env.__init__(self, **config)
        self.image = ImageF1()
        self.sensor = config["sensor"]

        # Image
        self.image_resizing = config["image_resizing"] / 100
        self.new_image_size = config["new_image_size"]
        self.raw_image = config["raw_image"]
        self.height = int(config["height_image"] * self.image_resizing)
        self.width = int(config["width_image"] * self.image_resizing)
        self.center_image = int(config["center_image"] * self.image_resizing)
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config["telemetry_mask"]
        # self.max_distance = config["max_distance"]
        self.poi = config["x_row"][0]
        self.image_center = None

        # States
        self.state_space = config["state_space"]
        if self.state_space == "spn":
            self.x_row = [i for i in range(1, int(self.height / 2) - 1)]
        else:
            self.x_row = config["x_row"]

        # Actions
        self.action_space = config["action_space"]
        self.actions = config["actions"]
        # self.beta_1 = config["beta_1"]
        # self.beta_0 = config["beta_0"]

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]
        # self.beta_1 = config["beta_1"]
        # self.beta_0 = config["beta_0"]

        # Others
        self.telemetry = config["telemetry"]

    def render(self, mode="human"):
        pass

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)

    def show_image(self, name, img, waitkey):
        window_name = f"{name}"
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    #################################################################################
    # Camera
    #################################################################################

    def image_msg_to_image(self, img, cv_image):
        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image

    def get_camera_info(self):
        image_data = None
        f1_image_camera = None
        success = False

        while image_data is None or success is False:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            if np.any(cv_image):
                success = True

        return f1_image_camera, cv_image

    #################################################################################
    # Center
    #################################################################################

    @staticmethod
    def get_center(lines):
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            # print(f"No lines detected in the image")
            return 0

    def processed_image(self, img):
        """
        Convert img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [mask[self.x_row[idx], :] for idx, x in enumerate(self.x_row)]
        centrals = list(map(self.get_center, lines))

        if self.telemetry_mask:
            mask_points = np.zeros((self.height, self.width), dtype=np.uint8)
            for idx, point in enumerate(centrals):
                cv2.line(
                    mask_points,
                    (int(point), int(self.x_row[idx])),
                    (int(point), int(self.x_row[idx])),
                    (255, 255, 255),
                    thickness=3,
                )
            cv2.imshow("MASK", mask_points[image_middle_line:])
            cv2.waitKey(3)

        return centrals

    def calculate_observation(self, state: list) -> list:
        """
        This is original Nacho's version.
        I have other version. See f1_env_ddpg_camera.py
        """
        # normalize = 40

        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.center_image - x) / self.pixel_region) + 1)
            # final_state.append(int(x / self.pixel_region) + 1)

        return final_state

    #################################################################################
    # Rewards
    #################################################################################

    def calculate_reward(self, error: float) -> float:

        d = np.true_divide(error, self.center_image)
        reward = np.round(np.exp(-d), 4)

        return reward

    def rewards_discrete(self, center):
        done = False
        if center > 0.9:
            done = True
            reward = self.rewards["penal"]
        elif 0 <= center <= 0.2:
            reward = self.rewards["from_0_to_02"]
        elif 0.2 < center <= 0.4:
            reward = self.rewards["from_02_to_04"]
        else:
            reward = self.rewards["from_others"]

        return reward, done

    def reward_v_w_center_linear(self, vel_cmd, center):
        """
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = -B_1 * Max V
        B_1 = -(W Max / (V Max - V Min))

        w target = B_0 + B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward + center))) where Max value = 1

        Args:
            linear and angular velocity
            center

        Returns: reward
        """
        done = False
        if center > 0.9:
            done = True
            reward = self.rewards["penal"]
        else:
            w_target = self.beta_0 + (self.beta_1 * abs(vel_cmd.linear.x))
            error = abs(w_target - abs(vel_cmd.angular.z))
            reward = 1 / math.exp(error + center)

        return reward, done

    def reward_v_w_center_linear_first_formula(self, vel_cmd, center):
        """
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = -B_1 * Max V
        B_1 = -(W Max / (V Max - V Min))

        w target = B_0 + B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward)/sqrt(center^2+0.001)

        Args:
            linear and angular velocity
            center

        Returns: reward
        """
        done = False
        if center > 0.9:
            done = True
            reward = self.rewards["penal"]
        else:
            num = 0.001
            w_target = self.beta_0 + (self.beta_1 * abs(vel_cmd.linear.x))
            error = abs(w_target - abs(vel_cmd.angular.z))
            reward = 1 / math.exp(error)
            reward = reward / math.sqrt(
                pow(center, 2) + num
            )  # Maximize near center and avoid zero in denominator

        return reward

    def reward_v_w_center_linear_second_formula(self, vel_cmd, center):
        """
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
        """
        num = 10
        w_target = self.beta_0 + (self.beta_1 * np.abs(vel_cmd.linear.x))
        error = np.abs(w_target - np.abs(vel_cmd.angular.z))
        reward = num * (1 / np.exp(error))
        reward = reward / (
            center + 0.01
        )  # Maximize near center and avoid zero in denominator
        return reward

    #################################################################################
    # image preprocessing
    #################################################################################

    def image_original_resizing_32x32(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_resized = cv2.resize(img_sliced, (32, 32), cv2.INTER_AREA)
        return img_resized

    def image_preprocessing_black_white_original_size(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY)
        mask_black_White_3D = np.expand_dims(mask, axis=2)

        return mask_black_White_3D

    def image_preprocessing_black_white_32x32(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY)
        mask_black_white_32x32 = cv2.resize(mask, (32, 32), cv2.INTER_AREA)
        mask_black_white_32x32 = np.expand_dims(mask_black_white_32x32, axis=2)

        return mask_black_white_32x32

    def image_preprocessing_gray_32x32(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        img_gray_3D = cv2.resize(img_proc, (32, 32), cv2.INTER_AREA)
        img_gray_3D = np.expand_dims(img_gray_3D, axis=2)

        return img_gray_3D

    def image_preprocessing_raw_original_size(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        return img_sliced

    def image_preprocessing_color_quantization_original_size(self, img):
        n_colors = 3
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        img_sliced = np.array(img_sliced, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img_sliced.shape)
        image_array = np.reshape(img_sliced, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0, n_samples=50)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)

        return kmeans.cluster_centers_[labels].reshape(w, h, -1)

    def image_preprocessing_color_quantization_32x32x1(self, img):
        n_colors = 3
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        img_sliced = np.array(img_sliced, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img_sliced.shape)
        image_array = np.reshape(img_sliced, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0, n_samples=500)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        im_reshape = kmeans.cluster_centers_[labels].reshape(w, h, -1)
        im_resize32x32x1 = np.expand_dims(np.resize(im_reshape, (32, 32)), axis=2)

        return im_resize32x32x1

    def image_preprocessing_reducing_color_PIL_original_size(self, img):
        num_colors = 4
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        array2pil = im.fromarray(img_sliced)
        array2pil_reduced = array2pil.convert(
            "P", palette=im.ADAPTIVE, colors=num_colors
        )
        pil2array = np.expand_dims(np.array(array2pil_reduced), 2)
        return pil2array

    def image_callback(self, image_data):
        self.image_raw_from_topic = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

    #################################################################################
    # reset
    #################################################################################

    def reset(self):
        """
        Main reset. Depending of:
        - sensor
        - states: images or simplified perception (sp)

        """
        # ic(sensor)
        if self.sensor == "camera":
            return self.reset_camera()

    def reset_camera(self):
        if self.alternate_pose:
            self._gazebo_set_new_pose()
        else:
            self._gazebo_reset()

        self._gazebo_unpause()
        # get image from sensor camera
        f1_image_camera, cv_image = self.get_camera_info()
        self._gazebo_pause()
        # calculating State
        # If image as observation
        if self.state_space == "image":
            # state = np.array(cv_image)
            # state = np.array(
            #    self.image_preprocessing_black_white_32x32(f1_image_camera.data)
            # )
            state = np.array(self.image_original_resizing_32x32(f1_image_camera.data))

            state_size = state.shape
        # ...or simplified perception as state
        else:
            points = self.processed_image(f1_image_camera.data)
            state = self.calculate_observation(points)
            state_size = len(state)

        return state, state_size

    def reset_camera_old(self):
        if self.alternate_pose:
            self._gazebo_set_new_pose()
        else:
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        image_data = None
        f1_image_camera = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            # f1_image_camera = cv_image
            if np.any(cv_image):
                success = True

        # veamos que center me da en el reset()
        points = self.processed_image(f1_image_camera.data)
        self._gazebo_pause()

        if self.state_space == "image":
            state = np.array(cv_image)
            state_size = state.shape
            return state, state_size
        else:
            state = self.calculate_observation(points)
            state_size = len(state)
            return state, state_size

    #################################################################################
    # step
    #################################################################################
    def step(self, action):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)
        # get image from sensor camera
        f1_image_camera, cv_image = self.get_camera_info()
        self._gazebo_pause()

        ######### center
        points = self.processed_image(f1_image_camera.data)
        if self.state_space == "spn":
            self.point = points[self.poi]
        else:
            self.point = points[0]
        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        ########## State
        if self.state_space == "image":
            state = np.array(cv_image)
        else:
            state = self.calculate_observation(points)

        ########## calculating Rewards
        if self.reward_function == "linear_follow_line":
            reward, done = self.reward_v_w_center_linear(vel_cmd, center)
        else:
            reward, done = self.rewards_discrete(center)

        return state, reward, done, {}

    def step_old(self, action):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        # Get camera info
        image_data = None
        f1_image_camera = None
        while image_data is None:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            # now resizing the image
            cv_image = cv2.resize(
                cv_image,
                (
                    int(cv_image.shape[1] * self.image_resizing / 100),
                    int(cv_image.shape[0] * self.image_resizing / 100),
                ),
                cv2.INTER_AREA,
            )

        self._gazebo_pause()
        points = self.processed_image(f1_image_camera.data)
        if self.state_space == "spn":
            self.point = points[self.poi]
        else:
            self.point = points[0]

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        # state = self.calculate_observation(points)
        # state = np.array(cv_image)
        if self.state_space == "image":
            state = np.array(cv_image)
        else:
            # state = self.calculate_observation(points)
            state = self.calculate_observation(points)

        done = False
        # calculate reward
        if center > 0.9:
            done = True
            reward = self.rewards["from_done"]
        else:
            reward = self.rewards_discrete(center)

        return state, reward, done, {}


#######################################################################
#######################################################################


class DQNF1FollowLaneEnvGazebo(F1Env):
    def __init__(self, **config):

        F1Env.__init__(self, **config)
        self.image = ImageF1()
        self.image_raw_from_topic = None
        self.f1_image_camera = None
        self.sensor = config["sensor"]

        # Image
        self.image_resizing = config["image_resizing"] / 100
        self.new_image_size = config["new_image_size"]
        self.raw_image = config["raw_image"]
        self.height = int(config["height_image"] * self.image_resizing)
        self.width = int(config["width_image"] * self.image_resizing)
        self.center_image = int(config["center_image"] * self.image_resizing)
        self.right_lane_center_image = int(
            (config["center_image"] + (config["center_image"] // 2))
            * self.image_resizing
        )
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config["telemetry_mask"]
        # self.max_distance = config["max_distance"]
        self.poi = config["x_row"][0]
        self.image_center = None

        # States
        self.state_space = config["state_space"]
        if self.state_space == "spn":
            self.x_row = [i for i in range(1, int(self.height / 2) - 1)]
        else:
            self.x_row = config["x_row"]

        # Actions
        self.action_space = config["action_space"]
        self.actions = config["actions"]
        # self.beta_1 = config["beta_1"]
        # self.beta_0 = config["beta_0"]

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]
        # self.beta_1 = config["beta_1"]
        # self.beta_0 = config["beta_0"]

        # Others
        self.telemetry = config["telemetry"]

    def render(self, mode="human"):
        pass

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)

    def show_image(self, name, img, waitkey):
        window_name = f"{name}"
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    #################################################################################
    # Camera
    #################################################################################

    def image_msg_to_image(self, img, cv_image):
        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image

    def get_camera_info(self):
        image_data = None
        f1_image_camera = None
        success = False

        while image_data is None or success is False:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            if np.any(cv_image):
                success = True

        return f1_image_camera, cv_image

    #################################################################################
    # Center
    #################################################################################

    @staticmethod
    def get_center(lines):
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            # print(f"No lines detected in the image")
            return 0

    def processed_image(self, img):
        """
        Convert img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [mask[self.x_row[idx], :] for idx, x in enumerate(self.x_row)]
        centrals = list(map(self.get_center, lines))

        return centrals

    def calculate_observation(self, state: list) -> list:
        """
        This is original Nacho's version.
        I have other version. See f1_env_ddpg_camera.py
        """
        # normalize = 40

        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.center_image - x) / self.pixel_region) + 1)
            # final_state.append(int(x / self.pixel_region) + 1)

        return final_state

    #################################################################################
    # Rewards
    #################################################################################

    def calculate_reward(self, error: float) -> float:

        d = np.true_divide(error, self.center_image)
        reward = np.round(np.exp(-d), 4)

        return reward

    def rewards_discrete_follow_right_lane(self, center):
        done = False
        if 0.75 > center >= 0.25:
            reward = self.rewards["from_10"]
        elif (1.06 > center >= 0.75) or (0.25 > center >= -0.06):
            reward = self.rewards["from_02"]
        elif (1.37 > center >= 1.06) or (-0.06 > center >= -0.37):
            reward = self.rewards["from_01"]
        else:
            done = True
            reward = self.rewards["penal"]

        return reward, done

    def rewards_discrete(self, center):
        done = False
        if center > 0.9:
            done = True
            reward = self.rewards["penal"]
        elif 0 <= center <= 0.2:
            reward = self.rewards["from_0_to_02"]
        elif 0.2 < center <= 0.4:
            reward = self.rewards["from_02_to_04"]
        else:
            reward = self.rewards["from_others"]

        return reward, done

    def reward_v_w_center_linear(self, vel_cmd, center):
        """
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = -B_1 * Max V
        B_1 = -(W Max / (V Max - V Min))

        w target = B_0 + B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward + center))) where Max value = 1

        Args:
            linear and angular velocity
            center

        Returns: reward
        """
        done = False
        if center > 0.9:
            done = True
            reward = self.rewards["penal"]
        else:
            w_target = self.beta_0 + (self.beta_1 * abs(vel_cmd.linear.x))
            error = abs(w_target - abs(vel_cmd.angular.z))
            reward = 1 / math.exp(error + center)

        return reward, done

    def reward_v_w_center_linear_first_formula(self, vel_cmd, center):
        """
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = -B_1 * Max V
        B_1 = -(W Max / (V Max - V Min))

        w target = B_0 + B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward)/sqrt(center^2+0.001)

        Args:
            linear and angular velocity
            center

        Returns: reward
        """
        done = False
        if center > 0.9:
            done = True
            reward = self.rewards["penal"]
        else:
            num = 0.001
            w_target = self.beta_0 + (self.beta_1 * abs(vel_cmd.linear.x))
            error = abs(w_target - abs(vel_cmd.angular.z))
            reward = 1 / math.exp(error)
            reward = reward / math.sqrt(
                pow(center, 2) + num
            )  # Maximize near center and avoid zero in denominator

        return reward

    def reward_v_w_center_linear_second_formula(self, vel_cmd, center):
        """
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
        """
        num = 10
        w_target = self.beta_0 + (self.beta_1 * np.abs(vel_cmd.linear.x))
        error = np.abs(w_target - np.abs(vel_cmd.angular.z))
        reward = num * (1 / np.exp(error))
        reward = reward / (
            center + 0.01
        )  # Maximize near center and avoid zero in denominator
        return reward

    #################################################################################
    # image preprocessing
    #################################################################################

    def image_original_resizing_32x32(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_resized = cv2.resize(img_sliced, (32, 32), cv2.INTER_AREA)
        return img_resized

    def image_preprocessing_black_white_original_size(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY)
        mask_black_White_3D = np.expand_dims(mask, axis=2)

        return mask_black_White_3D

    def image_preprocessing_black_white_32x32(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY)
        mask_black_white_32x32 = cv2.resize(mask, (32, 32), cv2.INTER_AREA)

        self.show_image("mask", mask, 1)
        self.show_image("mask", mask_black_white_32x32, 3)

        mask_black_white_32x32 = np.expand_dims(mask_black_white_32x32, axis=2)

        return mask_black_white_32x32

    def image_preprocessing_gray_32x32(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        img_gray_3D = cv2.resize(img_proc, (32, 32), cv2.INTER_AREA)
        img_gray_3D = np.expand_dims(img_gray_3D, axis=2)

        return img_gray_3D

    def image_preprocessing_raw_original_size(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        return img_sliced

    def image_preprocessing_color_quantization_original_size(self, img):
        n_colors = 3
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        img_sliced = np.array(img_sliced, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img_sliced.shape)
        image_array = np.reshape(img_sliced, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0, n_samples=50)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)

        return kmeans.cluster_centers_[labels].reshape(w, h, -1)

    def image_preprocessing_color_quantization_32x32x1(self, img):
        n_colors = 3
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        img_sliced = np.array(img_sliced, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img_sliced.shape)
        image_array = np.reshape(img_sliced, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0, n_samples=500)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        im_reshape = kmeans.cluster_centers_[labels].reshape(w, h, -1)
        im_resize32x32x1 = np.expand_dims(np.resize(im_reshape, (32, 32)), axis=2)

        return im_resize32x32x1

    def image_preprocessing_reducing_color_PIL_original_size(self, img):
        num_colors = 4
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        array2pil = im.fromarray(img_sliced)
        array2pil_reduced = array2pil.convert(
            "P", palette=im.ADAPTIVE, colors=num_colors
        )
        pil2array = np.expand_dims(np.array(array2pil_reduced), 2)
        return pil2array

    def image_callback(self, image_data):
        self.image_raw_from_topic = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

    #################################################################################
    # reset
    #################################################################################

    def reset(self):
        """
        Main reset. Depending of:
        - sensor
        - states: images or simplified perception (sp)

        """
        # ic(sensor)
        if self.sensor == "camera":
            return self.reset_camera()

    def reset_camera(self):
        self._gazebo_reset()
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_follow_rigth_lane()
        else:
            self._gazebo_set_fix_pose_f1_follow_right_lane()

        self._gazebo_unpause()
        # get image from sensor camera
        f1_image_camera, cv_image = self.get_camera_info()
        self._gazebo_pause()
        # calculating State
        # If image as observation
        if self.state_space == "image":
            # state = np.array(cv_image)
            state = np.array(
                self.image_preprocessing_black_white_32x32(f1_image_camera.data)
            )
            # state = np.array(self.image_original_resizing_32x32(f1_image_camera.data))

            state_size = state.shape
        # ...or simplified perception as state
        else:
            points = self.processed_image(f1_image_camera.data)
            state = self.calculate_observation(points)
            state_size = len(state)

        return state, state_size

    def reset_camera_old(self):
        if self.alternate_pose:
            self._gazebo_set_new_pose()
        else:
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        image_data = None
        f1_image_camera = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            # f1_image_camera = cv_image
            if np.any(cv_image):
                success = True

        # veamos que center me da en el reset()
        points = self.processed_image(f1_image_camera.data)
        self._gazebo_pause()

        if self.state_space == "image":
            state = np.array(cv_image)
            state_size = state.shape
            return state, state_size
        else:
            state = self.calculate_observation(points)
            state_size = len(state)
            return state, state_size

    #################################################################################
    # step
    #################################################################################
    def step(self, action):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)
        # get image from sensor camera
        f1_image_camera, cv_image = self.get_camera_info()
        self._gazebo_pause()

        ######### center
        points_in_red_line = self.processed_image(f1_image_camera.data)
        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]
        right_lane_center = float(self.right_lane_center_image - self.point) / (
            float(self.width) // 2
        )

        ########## State
        if self.state_space == "image":
            state = np.array(
                self.image_preprocessing_black_white_32x32(f1_image_camera.data)
            )
        else:
            state = self.calculate_observation(points_in_red_line)

        ########## calculating Rewards
        if self.reward_function == "linear_follow_line":
            reward, done = self.reward_v_w_center_linear(vel_cmd, right_lane_center)
        elif self.reward_function == "discrete_follow_right_lane":
            reward, done = self.rewards_discrete_follow_right_lane(right_lane_center)
        else:
            reward, done = self.rewards_discrete(right_lane_center)

        print_messages(
            "in step()",
            right_lane_center_image=self.right_lane_center_image,
            points_in_red_line=points_in_red_line,
            point=self.point,
            right_lane_center=right_lane_center,
            reward=reward,
            done=done,
        )

        return state, reward, done, {}

    def step_old(self, action):
        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        # Get camera info
        image_data = None
        f1_image_camera = None
        while image_data is None:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=5
            )
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            # now resizing the image
            cv_image = cv2.resize(
                cv_image,
                (
                    int(cv_image.shape[1] * self.image_resizing / 100),
                    int(cv_image.shape[0] * self.image_resizing / 100),
                ),
                cv2.INTER_AREA,
            )

        self._gazebo_pause()
        points = self.processed_image(f1_image_camera.data)
        if self.state_space == "spn":
            self.point = points[self.poi]
        else:
            self.point = points[0]

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        # state = self.calculate_observation(points)
        # state = np.array(cv_image)
        if self.state_space == "image":
            state = np.array(cv_image)
        else:
            # state = self.calculate_observation(points)
            state = self.calculate_observation(points)

        done = False
        # calculate reward
        if center > 0.9:
            done = True
            reward = self.rewards["from_done"]
        else:
            reward = self.rewards_discrete(center)

        return state, reward, done, {}
