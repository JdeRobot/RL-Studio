import math

from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np
from PIL import Image as im
import rospy
from sensor_msgs.msg import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.gazebo.f1.image_f1 import ImageF1
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env


class DDPGF1FollowLineEnvGazebo(F1Env):
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
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config["telemetry_mask"]
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

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]
        self.beta_1 = config["beta_1"]
        self.beta_0 = config["beta_0"]

        # Others
        self.telemetry = config["telemetry"]

    def render(self, mode="human"):
        pass

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)

    def show_image_with_centroid(self, name, img, waitkey, centroid=False):
        window_name = f"{name}"
        img_centroid = img
        if centroid:
            hsv = cv2.cvtColor(img_centroid, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([10, 10, 10])
            upper_yellow = np.array([255, 255, 250])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            h, w, d = img.shape
            search_top = 3 * h // 4
            search_bot = search_top + 20
            mask[0:search_top, 0:w] = 0
            mask[search_bot:h, 0:w] = 0
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] // M["m00"])
                cy = int(M["m01"] // M["m00"])
                cv2.circle(img_centroid, (cx, cy), 20, (0, 0, 255), -1)

        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

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
        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.center_image - x) / self.pixel_region) + 1)

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

    #################################################################################
    # image preprocessing
    #################################################################################

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

    def reset_camera_subscriber(self):
        """
        ropsy.Subscriber()

        """
        if self.alternate_pose:
            pass
        else:
            self._gazebo_reset()

        self._gazebo_unpause()
        success = False
        while self.cv_image is None or success is False:
            # from https://stackoverflow.com/questions/57271100/how-to-feed-the-data-obtained-from-rospy-subscriber-data-into-a-variable
            rospy.Subscriber(
                "/F1ROS/cameraL/image_raw", Image, self.image_callback, queue_size=1
            )
            if np.any(self.cv_image):
                success = True
        f1_image_camera = cv2.resize(
            self.cv_image,
            (
                int(self.cv_image.shape[1] * self.image_resizing),
                int(self.cv_image.shape[0] * self.image_resizing),
            ),
            cv2.INTER_AREA,
        )
        points = self.processed_image(self.cv_image)
        self._gazebo_pause()
        if self.state_space == "image":
            if self.raw_image:
                state = np.array(
                    self.image_preprocessing_reducing_color_PIL_original_size(
                        f1_image_camera
                    )
                )
            else:
                state = np.array(
                    self.image_preprocessing_black_white_32x32(f1_image_camera)
                )

            state_size = state.shape
            return state, state_size
        else:
            state = self.calculate_observation(points)
            state_size = len(state)
            return state, state_size

    def reset_camera(self):
        if self.alternate_pose:
            self._gazebo_set_new_pose()
        else:
            self._gazebo_reset()

        self._gazebo_unpause()
        # get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()
        # calculating State
        # If image as observation
        if self.state_space == "image":
            if self.raw_image:
                state = np.array(
                    self.image_preprocessing_reducing_color_PIL_original_size(
                        f1_image_camera.data
                    )
                )
            else:
                # no raw image
                state = np.array(
                    self.image_preprocessing_black_white_32x32(f1_image_camera.data)
                )
            state_size = state.shape

        # simplified perception as observation
        else:
            points = self.processed_image(f1_image_camera.data)
            state = self.calculate_observation(points)
            state_size = len(state)

        return state, state_size

    #################################################################################
    # step
    #################################################################################

    def step(self, action):
        self._gazebo_unpause()
        vel_cmd = Twist()

        if self.action_space == "continuous":
            vel_cmd.linear.x = action[0][0]
            vel_cmd.angular.z = action[0][1]
        else:
            vel_cmd.linear.x = self.actions[action][0]
            vel_cmd.angular.z = self.actions[action][1]

        self.vel_pub.publish(vel_cmd)
        # get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()

        ######### center
        points = self.processed_image(f1_image_camera.data)
        if self.state_space == "spn":
            self.point = points[self.poi]
        else:
            self.point = points[0]
        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        ########## calculating State
        # If image as observation
        if self.state_space == "image":
            if self.raw_image:
                state = np.array(
                    self.image_preprocessing_reducing_color_PIL_original_size(
                        f1_image_camera.data
                    )
                )
            else:
                # no raw image
                state = np.array(
                    self.image_preprocessing_black_white_32x32(f1_image_camera.data)
                )

        # simplified perception as observation
        else:
            state = self.calculate_observation(points)

        ########## calculating Rewards
        if self.reward_function == "linear_follow_line":
            reward, done = self.reward_v_w_center_linear(vel_cmd, center)
        else:
            reward, done = self.rewards_discrete(center)

        return state, reward, done, {}


########################################################
#
#                   Follow LANE
#
########################################################


class DDPGF1FollowLaneEnvGazebo(F1Env):
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
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config["telemetry_mask"]
        self.poi = config["x_row"][0]
        self.image_center = None
        self.right_lane_center_image = config["center_image"] + (
            config["center_image"] // 2
        )
        self.lower_limit = config["lower_limit"]

        # States
        self.state_space = config["state_space"]
        if self.state_space == "spn":
            self.x_row = [i for i in range(1, int(self.height / 2) - 1)]
        else:
            self.x_row = config["x_row"]

        # Actions
        self.action_space = config["action_space"]
        self.actions = config["actions"]

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]
        # self.beta_1 = config["beta_1"]
        # self.beta_0 = config["beta_0"]

        # Others
        self.telemetry = config["telemetry"]

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

    def reset_camera_subscriber(self):
        """
        ropsy.Subscriber()

        """
        if self.alternate_pose:
            pass
        else:
            self._gazebo_reset()

        self._gazebo_unpause()
        success = False
        while self.cv_image is None or success is False:
            # from https://stackoverflow.com/questions/57271100/how-to-feed-the-data-obtained-from-rospy-subscriber-data-into-a-variable
            rospy.Subscriber(
                "/F1ROS/cameraL/image_raw", Image, self.image_callback, queue_size=1
            )
            if np.any(self.cv_image):
                success = True
        f1_image_camera = cv2.resize(
            self.cv_image,
            (
                int(self.cv_image.shape[1] * self.image_resizing),
                int(self.cv_image.shape[0] * self.image_resizing),
            ),
            cv2.INTER_AREA,
        )
        points = self.processed_image(self.cv_image)
        self._gazebo_pause()
        if self.state_space == "image":
            if self.raw_image:
                state = np.array(
                    self.image_preprocessing_reducing_color_PIL_original_size(
                        f1_image_camera
                    )
                )
            else:
                state = np.array(
                    self.image_preprocessing_black_white_32x32(f1_image_camera)
                )

            state_size = state.shape
            return state, state_size
        else:
            state = self.calculate_observation(points)
            state_size = len(state)
            return state, state_size

    def reset_camera(self):
        self._gazebo_reset()
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_follow_rigth_lane()
        else:
            self._gazebo_set_fix_pose_f1_follow_right_lane()

        self._gazebo_unpause()
        # get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()

        self._gazebo_pause()

        # calculating State
        # If image as observation
        if self.state_space == "image":
            if self.raw_image:
                state = np.array(
                    self.image_preprocessing_reducing_color_PIL_original_size(
                        f1_image_camera.data
                    )
                )
            else:
                # no raw image
                state = np.array(
                    self.image_preprocessing_black_white_32x32(f1_image_camera.data)
                )
            state_size = state.shape

        # simplified perception as observation
        else:
            centrals_in_pixels, centrals_normalized = self.calculate_centrals_lane(
                f1_image_camera.data
            )
            states = self.calculate_observation(centrals_in_pixels)
            state = [states[0]]
            state_size = len(state)

        return state, state_size

    #################################################################################
    # Get Camera Info
    #################################################################################

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

    def image_msg_to_image(self, img, cv_image):
        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image

    #################################################################################
    # image preprocessing
    #################################################################################
    def image_preprocessing_black_white_32x32(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_gray = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
        mask_black_white_32x32 = cv2.resize(mask, (32, 32), cv2.INTER_AREA)
        mask_black_white_32x32 = np.expand_dims(mask_black_white_32x32, axis=2)

        self.show_image_simple("mask32x32", mask_black_white_32x32, 5)
        return mask_black_white_32x32

    #################################################################################
    # Center
    #################################################################################

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

        self.show_image("mask", mask, 5, centrals_in_pixels, centrals_normalized)

        return centrals_in_pixels, centrals_normalized

    @staticmethod
    def get_center_right_lane(lines):
        try:
            # inversed line
            inversed_lane = [x for x in reversed(lines)]
            # cut off right blanks
            inv_index_right = np.argmin(inversed_lane)
            # cropped right blanks
            cropped_lane = inversed_lane[inv_index_right:]
            # cut off central line
            inv_index_left = np.argmax(cropped_lane)
            # get real lane index
            index_real_right = len(lines) - inv_index_right
            if inv_index_left == 0:
                index_real_left = 0
            else:
                index_real_left = len(lines) - inv_index_right - inv_index_left
            # get center lane
            center = (index_real_right - index_real_left) // 2
            center_lane = center + index_real_left

            return center_lane

        except ValueError:
            return 0

    def calculate_observation(self, state: list) -> list:
        """ """
        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.center_image - x) / self.pixel_region) + 1)

        return final_state

    def show_image(self, name, img, waitkey, centrals_in_pixels, centrals_normalized):
        window_name = f"{name}"

        for index, value in enumerate(self.x_row):
            cv2.putText(
                img,
                str(f"{centrals_in_pixels[index]} - [{centrals_normalized[index]}]"),
                (centrals_in_pixels[index], self.x_row[index]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    #################################################################################
    # step
    #################################################################################

    def step(self, action):
        self._gazebo_unpause()
        vel_cmd = Twist()

        if self.action_space == "continuous":
            vel_cmd.linear.x = action[0][0]
            vel_cmd.angular.z = action[0][1]
        else:
            vel_cmd.linear.x = self.actions[action][0]
            vel_cmd.angular.z = self.actions[action][1]

        self.vel_pub.publish(vel_cmd)
        # get image from sensor camera
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()

        ######### center
        centrals_in_pixels, centrals_normalized = self.calculate_centrals_lane(
            f1_image_camera.data
        )
        ########## calculating State
        # If image as observation
        if self.state_space == "image":
            if self.raw_image:
                state = np.array(
                    self.image_preprocessing_reducing_color_PIL_original_size(
                        f1_image_camera.data
                    )
                )
            else:
                # no raw image
                state = np.array(
                    self.image_preprocessing_black_white_32x32(f1_image_camera.data)
                )

        # simplified perception as observation
        else:
            states = self.calculate_observation(centrals_in_pixels)
            state = [states[0]]

        ########## calculating Rewards
        if self.reward_function == "linear_follow_line":
            reward, done = self.reward_v_w_center_linear(vel_cmd, center)
        else:
            reward, done = self.rewards_discrete_follow_right_lane(
                centrals_in_pixels, centrals_normalized
            )

        # print_messages(
        #    "in step()",
        #    self_x_row=self.x_row,
        #    centrals_in_pixels=centrals_in_pixels,
        #    centrals_normalized=centrals_normalized,
        #    reward=reward,
        #    done=done,
        #    states=states,
        #    state=state,
        # )

        # if done:
        #    print_messages(
        #        "========= in step() and done=True ========================",
        #        centrals_in_pixels=centrals_in_pixels,
        #        centrals_normalized=centrals_normalized,
        #    )

        return state, reward, done, {}

    #################################################################################
    # Rewards
    #################################################################################

    def rewards_discrete_follow_right_lane(
        self, centrals_in_pixels, centrals_normalized
    ):
        done = False
        # if (centrals_in_pixels[1] > 500 and centrals_in_pixels[2] > 500) or (
        #    centrals_in_pixels[2] > 500 and centrals_in_pixels[3] > 350
        # ):
        #    done = True
        #    reward = self.rewards["penal"]
        # else:

        # if (
        #    centrals_normalized[0] > 0.9
        #    or (centrals_normalized[1] > 0.7 and centrals_normalized[2] > 0.7)
        #    or (centrals_normalized[2] > 0.6 and centrals_normalized[3] >= 0.5)
        # ):
        if centrals_normalized[0] > 0.8:
            done = True
            reward = self.rewards["penal"]
        elif centrals_normalized[0] <= 0.2468:
            reward = self.rewards["from_10"]
        elif 0.25 < centrals_normalized[0] <= 0.5:
            reward = self.rewards["from_02"]
        else:
            reward = self.rewards["from_01"]

        return reward, done

    #################################################################################
    # Show Image
    #################################################################################

    def show_image_with_centroid(self, name, img, waitkey, centroid=False):
        window_name = f"{name}"
        img_centroid = img
        if centroid:
            hsv = cv2.cvtColor(img_centroid, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([10, 10, 10])
            upper_yellow = np.array([255, 255, 250])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            h, w, d = img.shape
            search_top = 3 * h // 4
            search_bot = search_top + 20
            mask[0:search_top, 0:w] = 0
            mask[search_bot:h, 0:w] = 0
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] // M["m00"])
                cy = int(M["m01"] // M["m00"])
                cv2.circle(img_centroid, (cx, cy), 20, (0, 0, 255), -1)

        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    #    def show_image(self, name, img, waitkey, centrals_in_pixels, centrals_normalized):
    #        window_name = f"{name}"
    #
    #        for index, value in enumerate(self.x_row):
    #            cv2.putText(
    #                img,
    #                str(f"{centrals_in_pixels[index]},{centrals_normalized[index]}"),
    #                (centrals_in_pixels[index], self.x_row[index]),
    #                cv2.FONT_HERSHEY_SIMPLEX,
    #                0.5,
    #                (255, 255, 255),
    #                1,
    #                cv2.LINE_AA,
    #            )
    #        cv2.imshow(window_name, img)
    #        cv2.waitKey(waitkey)

    def show_image_simple(self, name, img, waitkey):
        window_name = f"{name}"
        img_centroid = img
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

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

    #################################################################################
    # image preprocessing
    #################################################################################

    def image_preprocessing_black_white_original_size(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY)
        mask_black_White_3D = np.expand_dims(mask, axis=2)

        return mask_black_White_3D

    def image_preprocessing_black_white_32x32_bgr2hsv(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY)
        mask_black_white_32x32 = cv2.resize(mask, (32, 32), cv2.INTER_AREA)
        mask_black_white_32x32 = np.expand_dims(mask_black_white_32x32, axis=2)

        self.show_image_simple("mask32x32", mask_black_white_32x32, 5)
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
