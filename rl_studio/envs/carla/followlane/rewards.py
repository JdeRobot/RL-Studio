import math

import numpy as np


CND = {
    20: -0.07,
    30: -0.1,
    40: -0.13,
    50: -0.17,
    60: -0.2,
    70: -0.23,
    80: -0.26,
    90: -0.3,
    100: -0.33,
    110: -0.36,
    120: -0.4,
    130: -0.42,
    140: -0.46,
    150: -0.49,
    160: -0.52,
    170: -0.56,
    180: -0.59,
    190: -0.62,
    200: -0.65,
    210: -0.69,
    220: -0.72,
}
CPD = {
    20: 343,
    30: 353,
    40: 363,
    50: 374,
    60: 384,
    70: 394,
    80: 404,
    90: 415,
    100: 425,
    110: 436,
    120: 446,
    130: 456,
    140: 467,
    150: 477,
    160: 488,
    170: 498,
    180: 508,
    190: 518,
    200: 528,
    210: 540,
    220: 550,
}


class AutoCarlaRewards:
    def rewards_followlane_two_lines(self, errors_normalized, center_image, params):
        """
        Lines right and left, such as LaneDetector or Sergios segmentation
        N points of perception

        rewards in x = [0, 1] and y = [0, 10]
        10.099 - (1.1 /(0.1089 + 10e(-12 * dist)))
        """

        a = 10.099
        b = 1.1
        c = 0.1089
        d = 10
        e = 12

        rewards = []
        done = False
        for index, _ in enumerate(errors_normalized):
            dist = errors_normalized[index] - center_image
            rewards.append(a - (b / (c + d * math.exp(-e * abs(dist)))))

        function_reward = sum(rewards) / len(rewards)

        if function_reward < 0.5:
            done = True

        # TODO: remove next comments to add new variables in reward function
        # function_reward += params["velocity"] * 0.5
        # function_reward -= params["steering_angle"] * 1.02

        return function_reward, done

    def rewards_sigmoid_only_right_line(self, dist_normalized, ground_truth_normalized):
        """
        ONLY FOR PERCEPTION WITH RIGHT LINE

        rewards in x = [0, 1] and y = [0, 10]
        10.099 - (1.1 /(0.1089 + 10e(-12 * dist)))
        """

        a = 10.099
        b = 1.1
        c = 0.1089
        d = 10
        e = 12

        rewards = []
        done = False
        for index, _ in enumerate(dist_normalized):
            dist = dist_normalized[index] - ground_truth_normalized[index]
            rewards.append(a - (b / (c + d * math.exp(-e * abs(dist)))))

        function_reward = sum(rewards) / len(rewards)

        # TODO: remove next comments
        # function_reward += params["velocity"] * 0.5
        # function_reward -= params["steering_angle"] * 1.02

        dist_normaliz_mean = sum(dist_normalized) / len(dist_normalized)
        if (
            function_reward < 0.3
            or dist_normaliz_mean >= 0.16
            or dist_normaliz_mean <= -0.5
        ):  # distance of -0.8 to right, and 0.6 to left
            done = True
            function_reward = 0

        return function_reward, done

    def rewards_right_line_gazebo(self, dist_normalized, params):
        rewards = []
        done = False
        for index, _ in enumerate(dist_normalized):
            # if dist_normalized[index] > 0:
            #    dist_normalized[index] = -dist_normalized[index]
            if 0.65 >= abs(dist_normalized[index]) >= 0.25:
                rewards.append(10)
            elif (0.9 > abs(dist_normalized[index]) > 0.65) or (
                0.25 >= abs(dist_normalized[index]) > 0
            ):
                rewards.append(2)
            # elif 0.0 >= dist_normalized[index] > -0.2:
            #    rewards.append(0.1)
            else:
                rewards.append(0)
                # done = True

        function_reward = sum(rewards) / len(rewards)

        # TODO: remove next comments
        # function_reward += params["velocity"] * 0.5
        # function_reward -= params["steering_angle"] * 1.02

        if function_reward < 0.06:  # when distance = 0.8
            done = True

        return function_reward, done

    def rewards_right_line(self, dist_normalized, ground_truth_normalized, params):
        rewards = []
        done = False
        # ground_truth_values = [CND[value] for i, value in enumerate(x_row)]

        for index, _ in enumerate(dist_normalized):
            if 0.2 >= ground_truth_normalized[index] - dist_normalized[index] >= 0 or (
                0 > ground_truth_normalized[index] - dist_normalized[index] >= -0.2
            ):
                rewards.append(10)
            elif (
                0.4 >= ground_truth_normalized[index] - dist_normalized[index] > 0.2
            ) or (
                -0.2 >= ground_truth_normalized[index] - dist_normalized[index] > -0.4
            ):
                rewards.append(2)

            else:
                rewards.append(-10)
                # done = True

        function_reward = sum(rewards) / len(rewards)

        # TODO: remove next comments
        # function_reward += params["velocity"] * 0.5
        # function_reward -= params["steering_angle"] * 1.02

        # if function_reward < 0.5:
        if function_reward < -7:
            done = True

        return function_reward, done

    def rewards_followlane_error_center(self, error, params):
        """
        Center, taken from 2 lines: left and right
        1 point1 of perception
        """

        rewards = []
        done = False
        for i, _ in enumerate(error):
            if error[i] > 0.85:
                rewards.append(10)
            elif 0.85 >= error[i] > 0.45:
                rewards.append(2)
            # elif 0.45 >= error[i] > 0.1:
            #    rewards.append(0.1)
            else:
                rewards.append(-100)
                done = True

        function_reward = sum(rewards) / len(rewards)

        # TODO: remove next comments
        # function_reward += params["velocity"] * 0.5
        # function_reward -= params["steering_angle"] * 1.02

        return function_reward, done

    def rewards_followlane_error_center_n_points(self, error, params):
        """
        N points of perception
        """

        rewards = []
        done = False
        for i, _ in enumerate(error):
            if error[i] > 0.85:
                rewards.append(10)
            elif 0.85 >= error[i] > 0.45:
                rewards.append(2)
            # elif 0.45 >= error[i] > 0.1:
            #    rewards.append(0.1)
            else:
                rewards.append(0)
                # done = True

        function_reward = sum(rewards) / len(rewards)
        if function_reward < 0.5:
            done = True

        # TODO: remove next comments
        # function_reward += params["velocity"] * 0.5
        # function_reward -= params["steering_angle"] * 1.02

        return function_reward, done

    def rewards_followline_center(self, center, rewards):
        """
        original for Following Line
        """
        done = False
        if center > 0.9:
            done = True
            reward = rewards["penal"]
        elif 0 <= center <= 0.2:
            reward = rewards["from_10"]
        elif 0.2 < center <= 0.4:
            reward = rewards["from_02"]
        else:
            reward = rewards["from_01"]

        return reward, done

    def rewards_followlane_dist_v_angle(self, error, params):
        # rewards = []
        # for i,_ in enumerate(error):
        #    if (error[i] < 0.2):
        #        rewards.append(10)
        #    elif (0.2 <= error[i] < 0.4):
        #        rewards.append(2)
        #    elif (0.4 <= error[i] < 0.9):
        #        rewards.append(1)
        #    else:
        #        rewards.append(0)
        rewards = [0.1 / error[i] for i, _ in enumerate(error)]
        function_reward = sum(rewards) / len(rewards)
        function_reward += math.log10(params["velocity"])
        function_reward -= 1 / (math.exp(params["steering_angle"]))

        return function_reward

    ########################################################
    # So far only Carla exclusively
    #
    ########################################################

    @staticmethod
    def rewards_followlane_centerline(center, rewards):
        """
        works perfectly
        rewards in function of center of Line
        """
        done = False
        if 0.65 >= center > 0.25:
            reward = rewards["from_10"]
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = rewards["from_02"]
        elif 0 >= center > -0.9:
            reward = rewards["from_01"]
        else:
            reward = rewards["penal"]
            done = True

        return reward, done

    def rewards_followlane_v_centerline_step(self, vel_cmd, center, step, rewards):
        """
        rewards in function of velocity, angular v and center
        """

        done = False
        if 0.65 >= center > 0.25:
            reward = (rewards["from_10"] + vel_cmd.linear.x) - math.log(step)
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = (rewards["from_02"] + vel_cmd.linear.x) - math.log(step)
        elif 0 >= center > -0.9:
            # reward = (self.rewards["from_01"] + vel_cmd.linear.x) - math.log(step)
            reward = -math.log(step)
        else:
            reward = rewards["penal"]
            done = True

        return reward, done

    def rewards_followlane_v_w_centerline(
        self, vel_cmd, center, rewards, beta_1, beta_0
    ):
        """
        v and w are linear dependents, plus center to the eq.
        """

        w_target = beta_0 - (beta_1 * abs(vel_cmd.linear.x))
        w_error = abs(w_target - abs(vel_cmd.angular.z))
        done = False

        if abs(center) > 0.9 or center < 0:
            done = True
            reward = rewards["penal"]
        elif center >= 0:
            reward = (
                (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
            )  # add a constant to favor right lane
            # else:
            #    reward = (1 / math.exp(w_error)) + (math.exp(center))

        return reward, done

    def calculate_reward(self, error: float) -> float:
        d = np.true_divide(error, self.center_image)
        reward = np.round(np.exp(-d), 4)
        return reward

    def rewards_followline_v_w_centerline(
        self, vel_cmd, center, rewards, beta_1, beta_0
    ):
        """
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = B_1 * Max V
        B_1 = (W Max / (V Max - V Min))

        w target = B_0 - B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward + center))) where Max value = 1

        Args:
                linear and angular velocity
                center

        Returns: reward
        """

        # print_messages(
        #    "in reward_v_w_center_linear()",
        #    beta1=self.beta_1,
        #    beta0=self.beta_0,
        # )

        w_target = beta_0 - (beta_1 * abs(vel_cmd.linear.x))
        w_error = abs(w_target - abs(vel_cmd.angular.z))
        done = False

        if abs(center) > 0.9:
            done = True
            reward = rewards["penal"]
        elif center > 0:
            reward = (
                (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
            )  # add a constant to favor right lane
        else:
            reward = (1 / math.exp(w_error)) + (math.exp(center))

        return reward, done
