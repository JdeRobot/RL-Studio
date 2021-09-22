import random

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Twist
from gym import spaces
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

from rl_studio.agents.f1.settings import actions, envs_params
from rl_studio.envs import gazebo_env
from rl_studio.envs.f1.image_f1 import ImageF1

font = cv2.FONT_HERSHEY_COMPLEX
time_cycle = 80
error = 0
integral = 0
v = 0
w = 0
current = "recta"
time_cycle = 80
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
TH = 500
buf = np.ones(25)
V = 4
V_MULT = 2
v_mult = V_MULT

max_distance = 0.5


class ImageF1:
    def __init__(self):
        self.height = 3  # Image height [pixels]
        self.width = 3  # Image width [pixels]
        self.timeStamp = 0  # Time stamp [s] */
        self.format = ""  # Image format string (RGB8, BGR,...)
        self.data = np.zeros(
            (self.height, self.width, 3), np.uint8
        )  # The image data itself
        self.data.shape = self.height, self.width, 3

    def __str__(self):
        s = f"Image: \n   height: {self.height}\n   width: {self.width}"
        s = f"{s}\n   format: {self.format}\n   timeStamp: {self.timeStamp}"
        return f"{s}\n   data: {self.data}"


class GazeboF1ManualCameraEnv(gazebo_env.GazeboEnv):
    def __init__(self, **config):
        # Launch the simulation with the given launchfile name
        self.circuit = envs_params["simple"]
        gazebo_env.GazeboEnv.__init__(self, config["launch"])
        self.vel_pub = rospy.Publisher("/F1ROS/cmd_vel", Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.action_space = spaces.Discrete(
            len(actions)
        )  # actions  # spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        self.position = None
        self.start_pose = np.array(self.circuit["start_pose"])
        self._seed()

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
            print(f"/gazebo/pause_physics service call failed: {e}")

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
            print(f"/gazebo/reset_simulation service call failed: {e}")

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
    def image_msg_to_image(img, cv_image):

        image = ImageF1()
        image.width = img.width
        image.height = img.height
        image.format = "RGB8"
        image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        image.data = cv_image

        return image

    @staticmethod
    def collinear3(x1, y1, x2, y2, x3, y3):
        """
        Funcion que devuelve el un factor de 'colinearidad' entre dos segmentos.
        Cuanto se acercan a una recta dos segmentos.
        """
        l = 0
        l = np.abs((y1 - y2) * (x1 - x3) - (y1 - y3) * (x1 - x2))
        return l

    def detect(self, points):
        """
        Funcion que detecta si el coche se encuentra en una recta o una curva mediante colinearidad de segmentos.
        """
        global current
        global buf
        l2 = 0
        l1 = self.collinear3(
            points[0][1],
            points[0][0],
            points[1][1],
            points[1][0],
            points[2][1],
            points[2][0],
        )
        if l1 > TH:
            buf[0] = 0
            current = "curva"
        else:
            # buffer que se alimenta cada vez que se da una deteccion de rectas. Dada la naturaleza del
            # circuito hay muchos falsos positivos, por lo que esto otorga un umbral de seguridad.
            buf = np.roll(buf, 1)
            buf[0] = 1
            if np.all(buf == 1):
                current = "recta"
        return (l1, l2)

    def get_point(self, index, img):
        """
        Funcion que devuelve el punto medio de una serie de puntos blancos (para detectar el punto central de la línea)
        """
        mid = 0
        if np.count_nonzero(img[index]) > 0:
            left = np.min(np.nonzero(img[index]))
            right = np.max(np.nonzero(img[index]))
            mid = np.abs(left - right) / 2 + left
        return mid

    def get_image(self):
        image_data = None
        f1_image_camera = None
        while image_data is None:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", Image, timeout=10
            )
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)

        # image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=1)
        # cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        # f1_image_camera = self.image_msg_to_image(image_data, cv_image)

        return f1_image_camera.data

    def execute(self):
        global error
        global integral
        global current
        global v_mult
        global v
        global w
        red_upper = (0, 255, 255)
        red_lower = (0, 30, 30)  # default: (0, 255, 171)
        kernel = np.ones((8, 8), np.uint8)
        image = self.get_image()
        image_cropped = image[300:, :, :]
        image_blur = cv2.GaussianBlur(image_cropped, (27, 27), 0)
        image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
        image_mask = cv2.inRange(image_hsv, red_lower, red_upper)
        image_eroded = cv2.erode(image_mask, kernel, iterations=3)

        rows, cols = image_mask.shape
        rows = rows - 1

        alt = 0
        ff = cv2.reduce(image_mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        if np.count_nonzero(ff[:, 0]) > 0:
            alt = np.min(np.nonzero(ff[:, 0]))

        points = []
        for i in range(3):
            if i == 0:
                index = alt
            else:
                index = rows / (2 * i)
            points.append((self.get_point(index, image_mask), index))

        points.append((self.get_point(rows, image_mask), rows))

        l, l2 = self.detect(points)

        if current == "recta":
            kp = 0.001
            kd = 0.004
            ki = 0
            cv2.circle(image_mask, (0, cols / 2), 6, RED, -1)

            if image_cropped[0, cols / 2, 0] < 170 and v > 8:
                accel = -0.4
            else:
                accel = 0.3

            v_mult = v_mult + accel
            if v_mult > 6:
                v_mult = 6
        else:
            kp = 0.011  # 0.018
            kd = 0.011  # 0.011
            ki = 0
            v_mult = V_MULT

        new_error = cols / 2 - points[0][0]

        proportional = kp * new_error
        error_diff = new_error - error
        error = new_error
        derivative = kd * error_diff
        integral = integral + error
        integral = ki * integral

        w = proportional + derivative + integral
        # v = V * v_mult

        vel_cmd = Twist()
        vel_cmd.linear.x = 3
        vel_cmd.angular.z = w
        self.vel_pub.publish(vel_cmd)

        # image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2RGB)
        # cv2.circle(image_mask, points[0], 6, GREEN, -1)
        # cv2.circle(image_mask, points[1], 6, GREEN, -1)  # punto central rows/2
        # cv2.circle(image_mask, points[2], 6, GREEN, -1)
        # cv2.circle(image_mask, points[3], 6, GREEN, -1)
        # cv2.putText(image_mask, 'w: {:+.2f} v: {}'.format(w, v), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, MAGENTA, 2,
        #             cv2.LINE_AA)
        # cv2.putText(image_mask, 'collinearU: {} collinearD: {}'.format(l, l2), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             MAGENTA, 2, cv2.LINE_AA)
        # cv2.putText(image_mask, 'actual: {}'.format(current), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, MAGENTA, 2,
        #             cv2.LINE_AA)

        # cv2.imshow("Image window", image_mask)
        # cv2.waitKey(3)

    def finish_line(self):
        x, y = self.get_position()
        current_point = np.array([x, y])

        dist = (self.start_pose - current_point) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        # print(dist)
        if dist < max_distance:
            return True
        return False

    def render(self, mode="human"):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass
