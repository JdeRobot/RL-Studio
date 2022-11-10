from typing import Tuple

import cv2
import numpy as np
import rospy
import time

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gym import spaces
from sensor_msgs.msg import Image

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.agents.f1.settings import QLearnConfig
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env
from rl_studio.envs.gazebo.gazebo_utils import set_new_pose
from rl_studio.envs.gazebo.f1.image_f1 import ListenerCamera


class F1CameraEnv(F1Env):
    def __init__(self, **config):
        F1Env.__init__(self, **config)
        self.image = ListenerCamera("/F1ROS/cameraL/image_raw")
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(3)
        # len(self.actions)
        # )  # actions  # spaces.Discrete(3)  # F,L,R
        self.config = QLearnConfig()

    def render(self, mode="human"):
        pass

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)

    def image_msg_to_image(self, img, cv_image):

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
            print(f"No lines detected in the image")
            return 0

    def calculate_reward(self, error: float) -> float:

        d = np.true_divide(error, self.config.center_image)
        reward = np.round(np.exp(-d), 4)

        return reward

    def processed_image(self, img: Image) -> list:
        """
        - Convert img to HSV.
        - Get the image processed.
        - Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """
        img_sliced = img[240:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(
            img_proc, (0, 30, 30), (0, 255, 255)
        )  # default: 0, 30, 30 - 0, 255, 200
        # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [
            mask[self.config.x_row[idx], :] for idx, x in enumerate(self.config.x_row)
        ]
        centrals = list(map(self.get_center, lines))

        # if centrals[-1] == 9:
        #     centrals[-1] = center_image

        if self.config.telemetry_mask:
            mask_points = np.zeros(
                (self.config.height, self.config.width), dtype=np.uint8
            )
            for idx, point in enumerate(centrals):
                cv2.line(
                    img_proc,
                    (int(point), int(self.config.x_row[idx])),
                    (int(point), int(self.config.x_row[idx])),
                    (255, 255, 255),
                    thickness=3,
                )

            cv2.imshow("MASK + POINT", img_proc)
            cv2.waitKey(1)

        return centrals

    def calculate_observation(self, state: list) -> list:

        normalize = 40

        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.config.center_image - x) / normalize) + 1)

        return final_state

    def step(self, action) -> Tuple:

        self._gazebo_unpause()
        
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)
        
        # Get camera info
        f1_image_camera = None
        start = time.time()

        f1_image_camera = self.image.getImage()
        self.previous_image = f1_image_camera.data
        
        while np.array_equal(self.previous_image, f1_image_camera.data):
            if (time.time() - start) > 0.1:
                vel_cmd = Twist()
                vel_cmd.linear.x = 0
                vel_cmd.angular.z = 0
                self.vel_pub.publish(vel_cmd)
            f1_image_camera = self.image.getImage()
        
        end = time.time()
        #print(end - start)
        
        self._gazebo_pause()
        points = self.processed_image(f1_image_camera.data)
        state = self.calculate_observation(points)
        center = float(self.config.center_image - points[0]) / (
            float(self.config.width) // 2
        )

        done = False
        center = abs(center)

        if center > 0.9:
            done = True
        if not done:
            if 0 <= center <= 0.2:
                reward = 10
            elif 0.2 < center <= 0.4:
                reward = 2
            else:
                reward = 1
        else:
            reward = -100
        
        return state, reward, done, {}

    def reset(self):
        # === POSE ===
        if self.alternate_pose:
            pos_number = set_new_pose(self.circuit_positions_set)
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
            if f1_image_camera:
                success = True

        points = self.processed_image(f1_image_camera.data)
        state = self.calculate_observation(points)
        # reset_state = (state, False)

        self._gazebo_pause()

        return state

    def inference(self, action):
        self._gazebo_unpause()

        vel_cmd = Twist()
        vel_cmd.linear.x = self.config.ACTIONS_SET[action][0]
        vel_cmd.angular.z = self.config.ACTIONS_SET[action][1]
        self.vel_pub.publish(vel_cmd)

        image_data = rospy.wait_for_message(
            "/F1ROS/cameraL/image_raw", Image, timeout=1
        )
        cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        f1_image_camera = self.image_msg_to_image(image_data, cv_image)

        self._gazebo_pause()

        points = self.processed_image(f1_image_camera.data)
        state = self.calculate_observation(points)

        center = float(self.configcenter_image - points[0]) / (
            float(self.config.width) // 2
        )

        done = False
        center = abs(center)

        if center > 0.9:
            done = True

        return state, done

    def finish_line(self):
        x, y = self.get_position()
        current_point = np.array([x, y])

        dist = (self.start_pose - current_point) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        # print(dist)
        if dist < self.config.max_distance:
            return True
        return False


#####################################################################################
#####################################################################################


class QlearnF1FollowLaneEnvGazebo(F1Env):
    def __init__(self, **config):
        F1Env.__init__(self, **config)
        self.image = ImageF1()
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(3)
        # len(self.actions)
        # )  # actions  # spaces.Discrete(3)  # F,L,R
        self.config = QLearnConfig()

        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]

        # Image
        # self.image_resizing = config["image_resizing"] / 100
        # self.new_image_size = config["new_image_size"]
        # self.raw_image = config["raw_image"]
        self.height = config["height_image"]
        self.width = config["width_image"]
        self.center_image = config["center_image"]
        self.right_lane_center_image = config["center_image"] + (
            config["center_image"] // 2
        )
        self.num_regions = config["num_regions"]
        self.lower_limit = config["lower_limit"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        # States
        self.state_space = config["state_space"]
        if self.state_space == "spn":  # dont use this option with Qlearn
            self.x_row = [i for i in range(1, int(self.height / 2) - 1)]
        else:
            self.x_row = config["x_row"]

    def render(self, mode="human"):
        pass

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)

    #################################################################################
    # Image
    #################################################################################
    def show_image(self, name, img, waitkey, centrals_in_pixels):
        window_name = f"{name}"

        # draw line with center reference
        start_point_1 = (0, self.x_row[0] - 4)
        end_point_1 = (640, self.x_row[0] - 4)
        color = (0, 0, 255)
        thickness = 2
        cv2.line(img, start_point_1, end_point_1, color, thickness)
        start_point_2 = (0, self.x_row[1] - 4)
        end_point_2 = (640, self.x_row[1] - 4)
        cv2.line(img, start_point_2, end_point_2, color, thickness)
        start_point_3 = (0, self.x_row[2] - 4)
        end_point_3 = (640, self.x_row[2] - 4)
        cv2.line(img, start_point_3, end_point_3, color, thickness)

        cv2.putText(
            img,
            str(f"{centrals_in_pixels[0]}"),
            (centrals_in_pixels[0], self.x_row[0] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            str(f"{centrals_in_pixels[1]}"),
            (centrals_in_pixels[1], self.x_row[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            str(f"{centrals_in_pixels[2]}"),
            (centrals_in_pixels[2], self.x_row[2] + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    def image_msg_to_image(self, img, cv_image):

        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image

    def offset_center_line(self, img):
        image_middle_line = self.height // 2
        # reduce img from top to middle deleting no relevant info
        img_sliced = img[image_middle_line:]
        # convert img to 2-D black and white
        img_hsv = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        img_new_range = cv2.inRange(img_hsv, (0, 30, 30), (0, 255, 255))
        _, mask_bw = cv2.threshold(img_new_range, 240, 255, cv2.THRESH_BINARY)
        # init first img(480,160) to zeros
        first = np.zeros((mask_bw.shape[0], 160))
        # concatenate img(480,160) + mask_bw(480,640) = (480, 800)
        img_concatenated = np.concatenate((first, mask_bw), axis=1)
        # resize to (480, 640)
        img_right_offset = img_concatenated[:, :640]

        # get the offset line centrals
        lines = [
            img_right_offset[self.x_row[idx], :] for idx, x in enumerate(self.x_row)
        ]
        centrals = list(map(self.get_center, lines))

        # print_messages(
        #    "in offset_center_line()",
        #    img_right_offset_shape=img_right_offset.shape,
        # )

        self.show_image("mask_offset", img_right_offset, 5)
        return centrals

    #################################################################################
    # Center
    #################################################################################

    @staticmethod
    def get_center_old(lines):
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            return 0

    @staticmethod
    def get_center(lines):
        try:
            pos_final_linea_negra = np.argmin(lines) + 15
            carril_derecho_entero = lines[pos_final_linea_negra:]
            final_carril_derecho = np.argmin(carril_derecho_entero)
            lim_izq = pos_final_linea_negra
            lim_der = pos_final_linea_negra + final_carril_derecho

            punto_central_carril = (lim_der - lim_izq) // 2
            punto_central_absoluto = lim_izq + punto_central_carril
            return punto_central_absoluto

        except ValueError:
            return 0

    def processed_image(self, img):
        """
        detecting middle of the right lane

        """
        image_middle_line = self.height // 2
        # cropped image from second half to bottom line
        img_sliced = img[image_middle_line:]
        # convert to black and white mask
        # lower_grey = np.array([30, 32, 22])
        # upper_grey = np.array([128, 128, 128])
        img_gray = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        # mask = cv2.inRange(img_sliced, lower_grey, upper_grey)

        # img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        # line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))
        # _, : list[ndarray[Any, dtype[generic]]]: list[ndarray[Any, dtype[generic]]]: list[ndarray[Any, dtype[generic]]]mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [mask[self.x_row[idx], :] for idx, x in enumerate(self.x_row)]
        # added last line (239) only FOllowLane, to break center line in bottom
        # lines.append(mask[self.lower_limit, :])

        centrals_in_pixels = list(map(self.get_center, lines))
        centrals_normalized = [
            float(self.center_image - x) / (float(self.width) // 2)
            for _, x in enumerate(centrals_in_pixels)
        ]

        self.show_image("mask", mask, 5)

        return centrals_in_pixels, centrals_normalized

    def processed_image_old(self, img):
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
        # added last line (239) only FOllowLane, to break center line in bottom
        lines.append(mask[self.lower_limit, :])

        centrals = list(map(self.get_center, lines))
        self.show_image("mask", mask, 5)

        return centrals

    def calculate_observation(self, state: list) -> list:

        # normalize = 40

        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.center_image - x) / self.pixel_region) + 1)

        return final_state

    #################################
    # Central Lane
    #################################

    @staticmethod
    def get_center_right_lane(lines):
        try:
            # inversed line
            inversed_lane = lines[::-1]
            # cut off right blanks
            index_right = np.argmin(inversed_lane)
            # cropped right blanks
            cropped_lane = inversed_lane[index_right:]
            # cut off central line
            index_left = np.argmax(cropped_lane)
            # get real lane index
            index_real_right = len(lines) - index_right
            index_real_left = len(lines) - index_right - index_left
            # get center lane
            center_lane = (index_real_right + index_real_left) // 2

            if center_lane >= 600:
                center_lane = 320

            return center_lane

        except ValueError:
            return 0

    def calculate_centrals_lane(self, img):
        image_middle_line = self.height // 2
        # cropped image from second half to bottom line
        img_sliced = img[image_middle_line:]
        # convert to black and white mask
        # lower_grey = np.array([30, 32, 22])
        # upper_grey = np.array([128, 128, 128])
        img_gray = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
        # get Lines to work for
        lines = [mask[self.x_row[idx], :] for idx, _ in enumerate(self.x_row)]
        # added last line (239), to control center line in bottom
        lines.append(mask[self.lower_limit, :])

        centrals_in_pixels = list(map(self.get_center_right_lane, lines))
        centrals_normalized = [
            abs(float(self.center_image - x) / (float(self.width) // 2))
            for _, x in enumerate(centrals_in_pixels)
        ]

        self.show_image("mask", mask, 5, centrals_in_pixels)

        return centrals_in_pixels, centrals_normalized

    #################################
    # Rewards
    #################################

    def rewards_discrete_follow_right_lane_lowerlimit(self, center, center_bottom):
        done = False
        if center_bottom > 480:
            done = True
            reward = self.rewards["penal"]
        else:
            if center > 0.9:
                done = True
                reward = self.rewards["penal"]
            elif 0 <= center <= 0.2:
                reward = self.rewards["from_10"]
            else:
                # elif 0.2 < center <= 0.4:
                reward = self.rewards["from_02"]
            # else:
            #    reward = self.rewards["from_01"]

        return reward, done

    def rewards_discrete_follow_right_lane(
        self, centrals_normalized, centrals_in_pixels
    ):
        done = False
        if (centrals_in_pixels[1] > 500 and centrals_in_pixels[2] > 500) or (
            centrals_in_pixels[2] > 500 and centrals_in_pixels[3] > 350
        ):
            done = True
            reward = self.rewards["penal"]
        else:
            if centrals_normalized > 0.9:
                done = True
                reward = self.rewards["penal"]
            elif 0 <= centrals_normalized <= 0.2:
                reward = self.rewards["from_10"]
            # elif 0.2 < centrals_normalized <= 0.4:
            #    reward = self.rewards["from_02"]
            else:
                reward = self.rewards["from_01"]

        return reward, done

    #################################
    # Step
    #################################

    def step(self, action) -> Tuple:

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
        # image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=1)
        # cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        # f1_image_camera = self.image_msg_to_image(image_data, cv_image)

        self._gazebo_pause()

        ###### DESDE AQUI ############################
        ## center REAL image
        centrals_in_pixels, centrals_normalized = self.calculate_centrals_lane(
            f1_image_camera.data
        )
        # centers = [
        #    float(self.center_image - x) / (float(self.width) // 2)
        #    for _, x in enumerate(centers_real_img)
        # ]
        # print_messages(
        #    "in step()",
        # lower_limit=self.lower_limit,
        #    self_x_row=self.x_row,
        #    centrals_in_pixels=centrals_in_pixels,
        #    centrals_normalized=centrals_normalized,
        # )
        ## calculating State (only one state)
        states = self.calculate_observation(centrals_in_pixels)
        # we take state
        state = [states[0]]

        ## calculating Rewards
        reward, done = self.rewards_discrete_follow_right_lane(
            centrals_normalized[0], centrals_in_pixels
        )

        ########### HASTA AQUI ##################################
        # print_messages(
        #    "in step()",
        # lower_limit=self.lower_limit,
        #    self_x_row=self.x_row,
        #    centrals_in_pixels=centrals_in_pixels,
        #    centrals_normalized=centrals_normalized,
        #    reward=reward,
        #    done=done,
        #    states=states,
        #    state=state,
        # )

        return state, reward, done, {}

    #################################
    # Reset
    #################################

    def reset(self):
        self._gazebo_reset()
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_follow_rigth_lane()
        else:
            self._gazebo_set_fix_pose_f1_follow_right_lane()

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
            if f1_image_camera:
                success = True

        points, points_normalized = self.calculate_centrals_lane(f1_image_camera.data)
        # print_messages(
        #    "in reset()",
        # lower_limit=self.lower_limit,
        #    points=points,
        #    points_normalized=points_normalized,
        # )
        states = self.calculate_observation(points)
        # reset_state = (state, False)
        # we take state
        state = [states[0]]

        self._gazebo_pause()

        return state

    #################################
    # Others
    #################################

    def inference(self, action):
        self._gazebo_unpause()

        vel_cmd = Twist()
        vel_cmd.linear.x = self.config.ACTIONS_SET[action][0]
        vel_cmd.angular.z = self.config.ACTIONS_SET[action][1]
        self.vel_pub.publish(vel_cmd)

        image_data = rospy.wait_for_message(
            "/F1ROS/cameraL/image_raw", Image, timeout=1
        )
        cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        f1_image_camera = self.image_msg_to_image(image_data, cv_image)

        self._gazebo_pause()

        points = self.processed_image(f1_image_camera.data)
        state = self.calculate_observation(points)

        center = float(self.configcenter_image - points[0]) / (
            float(self.config.width) // 2
        )

        done = False
        center = abs(center)

        if center > 0.9:
            done = True

        return state, done

    def finish_line(self):
        x, y = self.get_position()
        current_point = np.array([x, y])

        dist = (self.start_pose - current_point) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        # print(dist)
        if dist < self.config.max_distance:
            return True
        return False
