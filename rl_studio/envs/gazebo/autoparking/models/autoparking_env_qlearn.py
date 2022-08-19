from cmath import cos
import math

from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np
import rospy
from statistics import mean
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan

from agents.utils import print_messages
from rl_studio.envs.gazebo.autoparking.image_f1 import ImageF1
from rl_studio.envs.gazebo.autoparking.models.autoparking_env import AutoparkingEnv


class QlearnAutoparkingEnvGazebo(AutoparkingEnv):
    def __init__(self, **config):

        AutoparkingEnv.__init__(self, **config)
        self.image = ImageF1()
        # set callback variable
        self.laser_from_topic = None
        self.sensor = config["sensor"]

        # Image

        # States
        self.state_space = config["state_space"]
        self.states = config["states"]
        if self.state_space == "sp_curb":
            self.poi = self.states["poi"]
            self.regions = self.states["regions"]
            self.pixels_cropping = self.states["pixels_cropping"]
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

    #######################################################################################

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

    def state_sp_curb_processing(self, image, poi, pixels_cropping, num_regions):
        height, width, _ = image.shape
        middle_height = height // 2
        center_image = width // 2
        # quartil = 200

        # cutting image (240, 400)
        img_cropping = image[
            middle_height:,
            center_image - pixels_cropping : center_image + pixels_cropping,
        ]

        # convert to B&W
        img_gray = cv2.cvtColor(img_cropping, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=2)

        # get columns
        new_height, new_width, _ = mask.shape
        # reference_columns = [1, new_width // 2, new_width - 1]
        reference_columns = [
            int(((new_width / poi) // 2) + (i * new_width / poi)) for i in range(poi)
        ]

        columns = [
            mask[:, reference_columns[idx], 0]
            for idx, _ in enumerate(reference_columns)
        ]

        # get distance from every column
        distances2bottom_pixels = []
        for idx, _ in enumerate(reference_columns):
            try:
                distances2bottom_pixels.append(
                    new_height - np.min(np.nonzero(columns[idx]))
                )
            except:
                distances2bottom_pixels.append(-100)

        # dist2bottom_normalized = [
        #        float((new_height - distances2bottom_pixels[i]) / new_height)
        #        for i, _ in enumerate(distances2bottom_pixels)
        # ]

        # calculate points in regions
        regions = [
            int((new_height - x) / num_regions) + 1 if x > 0 else 0
            for _, x in enumerate(distances2bottom_pixels)
        ]

        return regions, distances2bottom_pixels

    #####################################################################################################
    def image_preprocessing(self, img, type="b&w"):
        # img_sliced = img[self.image_middle_line :]
        img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_proc, 200, 255, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=2)

        return mask

    #####################################################################################################
    def curb_image_processing(self, image):
        height, width, _ = image.shape
        # middle_height = height // 2
        middle_height = 280
        center_image = width // 2
        quartil = 50

        # resizing image
        mask_curb = image[
            middle_height:, center_image - quartil : center_image + quartil
        ]
        # self.show_image("mask_curb 240x100", mask_curb, 100)
        new_height, new_width, _ = mask_curb.shape

        # columns with blanks: 1, center, width - 1
        reference_columns = [1, new_width // 2, new_width - 1]

        columns = [
            mask_curb[:, reference_columns[idx], 0]
            for idx, _ in enumerate(reference_columns)
        ]

        # print_messages(
        #    "en curb_image_processing()",
        #    image_shape=image.shape,
        #    mask_curb_shape=mask_curb.shape,
        # )

        return reference_columns, columns, new_height

    def curb_distances_to_bottom(self, reference_columns, columns, height):
        distances2bottom_pixels = []
        for idx, _ in enumerate(reference_columns):
            try:
                distances2bottom_pixels.append(
                    height - np.min(np.nonzero(columns[idx]))
                )
            except:
                distances2bottom_pixels.append(-100)

        # print(f"en curb_distances_to_bottom(): distances2bottom = {distances2bottom}")

        # convert pixels to [0,1]
        # diff = height - distances2bottom[1]
        # dist2bottom = (diff - distances2bottom[1]) / (diff + distances2bottom[1])
        # dist2bottom = float(height - distances2bottom) / float(height)
        dist2bottom_normalized = [
            float((height - distances2bottom_pixels[i]) / height)
            for i, _ in enumerate(distances2bottom_pixels)
        ]

        return dist2bottom_normalized, distances2bottom_pixels

    def curb_angle(self, distances_bottom, reference_columns):
        a = distances_bottom[2] - distances_bottom[0]
        b = reference_columns[2] - reference_columns[0]
        hipt = np.hypot(a, b)
        angle_in_degrees = math.degrees(
            np.arccos((b**2 + hipt**2 - a**2) / (2 * b * hipt))
        )
        angle_in_radians = np.arccos((b**2 + hipt**2 - a**2) / (2 * b * hipt))
        return angle_in_degrees, angle_in_radians

    def lines_image_preprocessing(self, image):
        height, width, _ = image.shape
        middle_height = height // 2
        # middle_height = 280
        center_image = width // 2
        quartil = 200

        # resizing image
        mask_lines = image[
            middle_height:, center_image - quartil : center_image + quartil
        ]
        new_height, new_width, _ = mask_lines.shape
        new_center_image = new_width // 2

        # self.show_image("mask_lines 240x400", mask_lines, 100)
        # self.show_image(
        #    "mask_lines 240x200 right side", mask_lines[:, new_center_image + 1 :], 1
        # )

        lines_row = [middle_height - i for i in range(5, middle_height, 40)]

        lines_left = [
            mask_lines[lines_row[idx], :new_center_image]
            for idx, x in enumerate(lines_row)
        ]

        lines_right = [
            mask_lines[lines_row[idx], new_center_image:, 0]
            for idx, x in enumerate(lines_row)
        ]

        # print_messages(
        #    "en lines_image_preprocessing()",
        #    image_shape=image.shape,
        #    mask_lines_shape=mask_lines.shape,
        #    lines_row=lines_row,
        #    new_center_image=new_center_image,
        # lines_left_0=lines_left[0],
        # lines_right_0=lines_right[0],
        # )

        return lines_row, lines_left, lines_right, new_center_image

    def distances_right_line_to_center(self, lines_right, lines_row):
        distancias2right_lane = []
        for i, _ in enumerate(lines_row):
            try:
                distancias2right_lane.append(np.min(np.nonzero(lines_right[i])))
            except:
                distancias2right_lane.append(0)

        return distancias2right_lane

    def distances_left_line_to_center(self, lines_left, lines_row, center_image):
        distancias2left_lane = []
        for i, _ in enumerate(lines_row):
            try:
                distancias2left_lane.append(
                    center_image - np.max(np.nonzero(lines_left[i]))
                )
            except:
                distancias2left_lane.append(0)

        return distancias2left_lane

    def ratio_left2right_lines(self, distances2right_line, distances2left_line):
        # ratios = [
        #    abs((x - y) / (x + y))
        #    for x, y in zip(distances2right_line, distances2left_line)
        # ]
        ratios = []
        for x, y in zip(distances2right_line, distances2left_line):
            try:
                ratios.append(abs((x - y) / (x + y)))
            except:
                ratios.append(0)

        avg = [ratios[x] for x in range(len(ratios)) if ratios[x] > 0]
        try:
            return mean(avg)
        except:
            return 0

    def params_parkingslot(self, img):
        # image = self.image_preprocessing(img, (img.shape[0], img.shape[1]))
        image = img

        # preprocessing image for curb distance to bottom
        reference_columns, columns, curb_new_height = self.curb_image_processing(
            self.image_preprocessing(img, "b&w")
        )

        # curb distance to bottom
        distances2bottom_norm, distances2bottom_pixels = self.curb_distances_to_bottom(
            reference_columns, columns, curb_new_height
        )

        # curb angle
        angle_in_degrees, angle_in_radians = self.curb_angle(
            distances2bottom_pixels, reference_columns
        )

        # preprocessing image for left-right lines
        (
            lines_row,
            lines_left,
            lines_right,
            new_center_image,
        ) = self.lines_image_preprocessing(self.image_preprocessing(image, "b&w"))

        # distances to right line
        distances2right_line = self.distances_right_line_to_center(
            lines_right, lines_row
        )

        # distances to left line
        distances2left_line = self.distances_left_line_to_center(
            lines_left, lines_row, new_center_image
        )

        # calculating ratio between left and right line
        ratio_l_r = self.ratio_left2right_lines(
            distances2right_line, distances2left_line
        )

        return (
            distances2bottom_norm,
            distances2bottom_pixels,
            angle_in_degrees,
            angle_in_radians,
            distances2right_line,
            distances2left_line,
            ratio_l_r,
        )

    #####################################################################################################
    # REWARDS camera
    #####################################################################################################

    def rewards_near_parkingspot_discrete(
        self, curb_dist, v, angle=None, ratio_l_r=None
    ):
        if curb_dist < 0.5 and v > 1:
            reward = 10
        elif (0.84 > curb_dist >= 0.5) and (1 >= v > 0.5):
            reward = 20
        elif (0.95 > curb_dist >= 0.84) and (0.5 >= v > 0):
            reward = 40
        else:
            reward = 1

        return reward

    def rewards_near_parkingspot(self, curb_dist, v, angle, ratio_l_r):
        # v_0_count = 0
        # penalty_reward = -100
        # goal_reward = 10_000

        # WRONG v = 0 and far from curb or overpassing curb
        if v == 0 and (curb_dist[1] < 0.6 or mean(curb_dist) >= 1):
            done = True
            reward = self.rewards["penal_reward"]
            # info = "out for v = 0 and far from curb or overpassing curb"
            print_messages(
                "rewards: done for v = 0 and far from curb or overpassing curb",
            )
        # WRONG left-right ratio unbalanced
        elif curb_dist[1] < 0.7 and ratio_l_r > 0.9:
            done = True
            reward = self.rewards["penal_reward"]
            # info = "out left-right ratio unbalanced"
            print_messages(
                "rewards: done left-right ratio unbalanced",
            )
        # WRONG curb angle very inclined
        elif angle > 10:
            done = True
            reward = self.rewards["penal_reward"]
            # info = "curb angle very inclined"
            print_messages(
                "rewards: done angle very inclined",
            )
        # WRONG out of margin
        elif curb_dist[1] > 0.83 and v != 0:
            done = True
            reward = self.rewards["penal_reward"]

        # done for distance
        elif 1 > curb_dist[1] > 0.83 and v == 0:
            done = True
            reward = self.rewards["goal_reward"]
            print_messages("rewards: GOAAAAALLL!!")

        # we must avoid v=0 for long time
        # elif v == 0 and (curb_dist[1] < 0.92 or curb_dist != -100):
        #    v_0_count += 1
        #    if v_0_count > 3:
        #        done = True
        #        reward = penalty_reward
        #    print_messages(
        #        "rewards: out for v=0 for long time",
        #    )

        # everything ok
        else:
            reward = self.rewards_near_parkingspot_discrete(
                curb_dist[1],
                v,
                angle,
                ratio_l_r,
            )
            # reward = self.rewards_near_parkingspot_discrete(
            #    curb_dist[1],
            #    ratio_l_r,
            # )
            done = False
            # info = "ok"

        return round(reward, 2), done

    #####################################################################################################
    # Rewards Simplified percepction

    #####################################################################################################

    def rewards_near_parkingspot_linear_sp(self, curb_dist, v, angle, ratio_l_r):
        vmax = 3.0
        vmin = -1
        dmax = 0.85
        dmin = 0
        B_1 = vmax / (dmax - dmin)
        B_0 = vmax
        # print(curb_dist)
        v_target = B_0 - (B_1 * curb_dist)
        error = abs(v - v_target)
        try:
            reward_near = (
                (1 / math.exp(error)) * math.cos(angle) * (1 / math.exp(ratio_l_r))
            )
        except:
            reward_near = 0

        # discretizing rewards for fast speed learning
        if 1 >= reward_near >= 0.9:
            reward = 1
        elif 0.9 > reward_near >= 0.75:
            reward = 0.5
        elif 0.75 > reward_near >= 0.5:
            reward = 0.2
        else:
            reward = 0

        return reward

    def rewards_near_parkingspot_discrete_sp(self, curb_dist, v):
        if curb_dist < 0.5 and v > 1:
            reward = 5
        elif (0.84 > curb_dist >= 0.5) and (1 >= v > 0.5):
            reward = 10
        # elif (0.95 > curb_dist >= 0.84) and (0.5 >= v > 0):
        #    reward = 40
        else:
            reward = 0

        # if (0.84 > curb_dist >= 0.5) and v > 1:
        #    reward = 20
        # elif (0.95 > curb_dist >= 0.84):
        #    reward = 40
        # else:
        #    reward = 1

        return reward

    def rewards_near_parkingspot_sp(self, curb_dist, v, angle, ratio_l_r):
        # v_0_count = 0
        # penalty_reward = -100
        # goal_reward = 10_000

        # WRONG out of margin
        if curb_dist[1] > 0.83 and v != 0:
            done = True
            reward = self.rewards["penal_reward"]

        # done for distance
        elif 1 > curb_dist[1] > 0.83 and v == 0:
            done = True
            reward = self.rewards["goal_reward"]
            print_messages("rewards: GOAAAAALLL!!")

        # everything ok
        else:
            reward = self.rewards_near_parkingspot_linear_sp(
                curb_dist[1],
                v,
                angle,
                ratio_l_r,
            )
            done = False

        return round(reward, 2), done

    #####################################################################################################
    # REWARDS laser
    #####################################################################################################

    def rewards_near_parkingspot_discrete_laser(self, state, v):
        if curb_dist < 0.5 and v > 1:
            reward = 5
        elif (0.84 > curb_dist >= 0.5) and (1 >= v > 0.5):
            reward = 10
        # elif (0.95 > curb_dist >= 0.84) and (0.5 >= v > 0):
        #    reward = 40
        else:
            reward = 0

        # if (0.84 > curb_dist >= 0.5) and v > 1:
        #    reward = 20
        # elif (0.95 > curb_dist >= 0.84):
        #    reward = 40
        # else:
        #    reward = 1

        return reward

    def rewards_near_parkingspot_laser(self, state, v):
        # v_0_count = 0
        # penalty_reward = -100
        # goal_reward = 10_000

        # WRONG out of margin
        if state == 0 and v != 0:
            done = True
            reward = self.rewards["penal_reward"]

        # done for distance
        elif state == 0 and v == 0:
            done = True
            reward = self.rewards["goal_reward"]
            print_messages("rewards: GOAAAAALLL!!")

        # everything ok
        else:
            try:
                reward = abs(1 / v)
            except:
                reward = 0.2
            done = False

        return round(reward, 2), done

    #####################################################################################################
    # LASER
    #####################################################################################################
    """
    @staticmethod
    def discrete_observation(data, new_ranges):

        discrete_ranges = []
        # min_range = 0.05
        done = False
        mod = len(data.ranges) / new_ranges
        # filter_data = data.ranges
        for i, __ in enumerate(data.ranges):
            if i % mod == 0:
                if filter_data[i] == float("Inf") or np.isinf(filter_data[i]):
                    # discrete_ranges.append(6)
                    discrete_ranges.append(int(data.range_max))
                elif np.isnan(filter_data[i]):
                    discrete_ranges.append(0)
                else:
                    discrete_ranges.append(int(filter_data[i]))
            if min_range > filter_data[i] > 0:
                print("Data ranges: {}".format(data.ranges[i]))
                done = True
                break

        return discrete_ranges, done
    """

    def clean_laser_beans(self, data):
        """inf or nan data converts = range_max = 10 m."""
        # lenght_data = data.ranges
        clean_beans = [
            int(data.range_max)
            if data.ranges[i] == float("Inf") or np.isinf(data.ranges[i])
            else int(data.range_max)  # 0
            if np.isnan(data.ranges[i])
            else data.ranges[i]
            for i, _ in enumerate(data.ranges)
        ]
        return clean_beans

    def regions_mins_laser_beans(self, data, new_ranges):
        """gets the min value of each segment wich comes from new_ranges
        default: segment = 3
        """
        clean_data = self.clean_laser_beans(data)
        # print_messages(
        #    "regions_mins_laser_beans()",
        #    clean_data=clean_data,
        # )
        discrete_states = [
            min(
                clean_data[
                    int(i * (len(clean_data) / new_ranges)) : int(
                        (i + 1) * (len(clean_data) / new_ranges) - 1
                    )
                ]
            )
            for i in range(new_ranges)
        ]
        return discrete_states

    def max_laser_range_ahead(self, data):
        """gets range value in front of the car.
        If Inf or nan = range.max
        """
        # print_messages(
        #    "max_laser_range_ahead",
        #    len_data_ranges=len(data.ranges),
        #    data=data,
        # )
        range_ahead = data.ranges[len(data.ranges) // 2]
        if (
            range_ahead == float("Inf")
            or np.isinf(range_ahead)
            or np.isnan(range_ahead)
        ):
            range_ahead = int(data.range_max)
        return range_ahead

    def get_laser_info_try(self):
        """read data from laser sensor"""
        laser_data = None
        success = False
        while laser_data is None or not success:
            try:
                laser_data = rospy.wait_for_message(
                    "/F1ROS/laserC/scan", LaserScan, timeout=5
                )
            finally:
                success = True

        return laser_data

    def get_laser_info_wait_for_message(self):
        """read data from laser sensor"""
        laser_data = None
        success = False
        # while laser_data is None and success is False:
        while success is False:
            laser_data = rospy.wait_for_message(
                "/F1ROS/laserC/scan", LaserScan, timeout=5
            )
            if laser_data.ranges and laser_data.angle_max > 0:
                success = True

        return laser_data

    def laser_callback(self, laser_data):
        self.laser_from_topic = laser_data

    def get_laser_info(self):
        """with callbacks"""
        # laser_data = None
        success = False
        # while laser_data is None and success is False:
        while success is False:
            rospy.Subscriber(
                "/F1ROS/laserC/scan", LaserScan, self.laser_callback, queue_size=1
            )
            if self.laser_from_topic:
                success = True

        return self.laser_from_topic

    def states_discretized(self, states):
        """if distance is:
        less than 1 meter: state 0
        between 1 and 3 meters: state 1
        more than 3 meters: state 2
        """
        states_discretized = [
            0 if states[i] < 1 else 1 if 3 >= states[i] >= 1 else 2
            for i, _ in enumerate(states)
        ]
        return states_discretized

    def euclidean_distance2parking(p1, p2):
        """
        calculates distance between parking spot (fixed) amd ego vehicle
        """
        x1, y1 = p1
        x2, y2 = p2
        return math.hypot(x2 - x1, y2 - y1)

    def euclidean_distance2parking_faster(p1, p2):
        """
        list comprehension - calculates distance between parking spot (fixed) amd ego vehicle
        """
        return math.sqrt(sum((x2 - x1) ** 2 for x1, x2 in zip(p1, p2)))

    def manhattan_distance(a, b):
        """
        calculates manhattan distance between parking spot and ego vehicle
        """
        return sum(abs(v1 - v2) for v1, v2 in zip(a, b))

    ######################################################################################################################
    # RESET
    # ####################################################################################

    def reset(self):
        if self.sensor == "camera":
            return self.reset_camera()
        elif self.sensor == "laser":
            return self.reset_laser()

    def reset_laser(self):
        # self._gazebo_set_new_pose()
        self._gazebo_reset()
        self._gazebo_set_new_pose()
        # self._gazebo_set_random_new_pose()

        self._gazebo_unpause()
        # read laser sensor data
        laser_data = self.get_laser_info()
        self._gazebo_pause()
        # print(laser_data)
        ## calculate min in every region
        state = self.regions_mins_laser_beans(laser_data, 5)
        # print(state)
        ## discretizing states= [0,1,2]
        states_discretized = self.states_discretized(state)

        ## State only distance ahead [1]
        ## calculate max distance in front of ego vehicle
        max_range_ahead = self.max_laser_range_ahead(laser_data)
        ## distance ahead discretized
        state = self.states_discretized([max_range_ahead])

        print_messages(
            "en reset_laser()",
            state=state,
            states_discretized=states_discretized,
            # len_data_ranges=len(laser_data.ranges),
            max_range_ahead=max_range_ahead,
            # distance_ahead=distance_ahead,
            # laser_data=laser_data,
        )
        return state

    def reset_camera(self):

        if self.alternate_pose:
            self._gazebo_set_random_new_pose()  # Mine, it works fine!!!
        else:
            self._gazebo_set_new_pose()
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        f1_image_camera, _ = self.get_camera_info()

        self._gazebo_pause()

        # Calculating STATE
        # we want state as image preprocessed
        if self.state_space == "sp_curb":
            state, _ = self.state_sp_curb_processing(
                f1_image_camera.data, self.poi, self.pixels_cropping, self.regions
            )
            state_size = len(state)

        return state, state_size

    #####################################################################################################
    # STEP
    #####################################################################################################
    def step(self, action):
        if self.sensor == "camera":
            return self.step_camera(action)
        elif self.sensor == "laser":
            return self.step_laser(action)

    def step_laser(self, action):

        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]

        ## publishing to Topic
        self.vel_pub.publish(vel_cmd)

        ## getting agent position
        x_agent, y_agent = self.get_position()
        # print_messages(
        #    "step()",
        #    x_agent=x_agent,
        #    y_agent=y_agent,
        # )

        ## Calculating STATE
        ## read laser sensor data
        laser_data = self.get_laser_info()
        self._gazebo_pause()

        # print_messages(
        #    "step()",
        #    laser_data=laser_data,
        # )
        ## calculate min in every region
        # state = self.regions_mins_laser_beans(laser_data, 5)
        ## discretizing states= [0,1,2]
        # states_discretized = self.states_discretized(state)
        ## calculate max distance in front of ego vehicle
        max_range_ahead = self.max_laser_range_ahead(laser_data)
        ## distance ahead discretized
        state = self.states_discretized([max_range_ahead])

        done = False
        ### reward & done
        reward, done = self.rewards_near_parkingspot_laser(
            state,
            vel_cmd.linear.x,
        )

        print_messages(
            "en step_laser()",
            state=state,
            v=vel_cmd.linear.x,
            x_agent=x_agent,
            y_agent=y_agent,
            #   states_discretized=states_discretized,
            # len_data_ranges=len(laser_data.ranges),
            max_range_ahead=max_range_ahead,
            # distance_ahead=distance_ahead,
            # laser_data=laser_data,
            reward=reward,
            done=done,
        )
        return state, reward, done, {}

    def step_camera(self, action):

        self._gazebo_unpause()
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]

        # publishing to Topic
        self.vel_pub.publish(vel_cmd)

        # Get camera info (original 480x640)
        f1_image_camera, _ = self.get_camera_info()
        self._gazebo_pause()

        # Calculating STATE
        # we want state as image preprocessed
        if self.state_space == "sp_curb":
            state, curb_distances2bottom_pixels_sp = self.state_sp_curb_processing(
                f1_image_camera.data, self.poi, self.pixels_cropping, self.regions
            )

        # calculating params parking lot
        (
            curb_distance2bottom_norm,
            curb_distance2bottom_pixels,
            curb_angle_degrees,
            curb_angle_in_radians,
            distances2right_line,
            distances2left_line,
            ratio_left2right_lines,
        ) = self.params_parkingslot(f1_image_camera.data)

        done = False
        ### reward & done
        reward, done = self.rewards_near_parkingspot_sp(
            curb_distance2bottom_norm,
            vel_cmd.linear.x,
            curb_angle_in_radians,
            ratio_left2right_lines,
        )

        # info = curb_distance2bottom_norm[1]

        print_messages(
            "in step()",
            # f1_image_camera_data=f1_image_camera.data.shape,
            v=vel_cmd.linear.x,
            w=vel_cmd.linear.z,
            # state_dims=state.shape,
            # poi=self.poi,
            # pixels_cropping=self.pixels_cropping,
            # regions=self.regions,
            # state=state,
            curb_distances2bottom_pixels_sp=curb_distances2bottom_pixels_sp,
            curb_distance2bottom_norm=curb_distance2bottom_norm,
            curb_distance2bottom_pixels=curb_distance2bottom_pixels,
            # curb_angle_degrees=curb_angle_degrees,
            # curb_angle_in_radians=curb_angle_in_radians,
            # distances2right_line=distances2right_line,
            # distances2left_line=distances2left_line,
            # ratio_left2right_lines=ratio_left2right_lines,
            # reward=reward,
            # done=done,
        )
        return state, reward, done, {}
