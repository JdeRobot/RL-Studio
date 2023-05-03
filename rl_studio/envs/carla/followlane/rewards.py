import math

import numpy as np


class AutoCarlaRewards:
    def rewards_right_line(self, dist_normalized, x_row, params):
        normalized_distance = {
            10: 0,
            20: 0,
            30: -0.1,
            60: -0.1,
            80: -0.2,
            100: -0.3,
            130: -0.5,
            180: -0.5,
            200: -0.5,
            230: -0.6,
        }
        ground_truth_values = [
            normalized_distance[value] for i, value in enumerate(x_row)
        ]

        rewards = []
        done = False
        for index, _ in enumerate(dist_normalized):
            # if dist_normalized[index] > 0:
            #    dist_normalized[index] = -dist_normalized[index]
            if 0.2 >= abs(dist_normalized[index] - ground_truth_values[index]) >= 0:
                rewards.append(10)
            elif 0.4 >= abs(dist_normalized[index] - ground_truth_values[index]) > 0.2:
                rewards.append(2)
            elif 0.8 >= abs(dist_normalized[index] - ground_truth_values[index]) > 0.4:
                rewards.append(0.1)
            else:
                rewards.append(-100)
                done = True

        function_reward = sum(rewards) / len(rewards)

        # TODO: remove next comments
        # function_reward += params["velocity"] * 0.5
        # function_reward -= params["steering_angle"] * 1.02

        return function_reward, done

    def rewards_easy(self, error, params):
        rewards = []
        done = False
        for i, _ in enumerate(error):
            if error[i] > 0.85:
                rewards.append(10)
            elif 0.85 >= error[i] > 0.45:
                rewards.append(2)
            elif 0.45 >= error[i] > 0.1:
                rewards.append(0.1)
            else:
                rewards.append(-100)
                done = True

        function_reward = sum(rewards) / len(rewards)

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
