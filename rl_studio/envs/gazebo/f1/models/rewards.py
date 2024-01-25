import math

import numpy as np


class F1GazeboRewards:
    def __init__(self, **config):
        self.reward_function_tuning = config.get("reward_function_tuning")
        self.punish_zig_zag_value = config.get("punish_zig_zag_value")
        self.punish_ineffective_vel = config.get("punish_ineffective_vel")
        self.beta_1 = config.get("beta_1")

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

    def rewards_followline_velocity_center(self, v, w, state, range_v, range_w):
        """
        original for Following Line
        """
        # we reward proximity to the line
        p_reward1, done1 = self.reward_proximity(state[-2])
        p_reward2, done2 = self.reward_proximity(state[-4])
        p_reward3, done3 = self.reward_proximity(state[-5])
        p_reward = (p_reward1 + p_reward2 + p_reward3)/3
        done = done1 and done2 and done3

        v_norm = self.normalize_range(v, range_v[0], range_v[1])
        v_reward = v_norm * pow(p_reward, 2)
        beta = self.beta_1
        reward = (beta * p_reward) + ((1 - beta) * v_reward)

        # penalizing steering to avoid zig-zag
        w_punish = self.normalize_range(abs(w), 0, abs(range_w[1])) * self.punish_zig_zag_value
        reward = reward - (reward * w_punish)

        # penalizing accelerating on bad positions
        v_punish = reward * (1 - p_reward) * v_norm * self.punish_ineffective_vel
        reward = reward - v_punish

        return reward, done

    def normalize_range(self, num, a, b):
        return (num - a) / (b - a)

    def reward_proximity(self, state):
        # sigmoid_pos = self.sigmoid_function(0, 1, state)
        if abs(state) > 0.7:
            return 0, True
        else:
            if self.reward_function_tuning == "sigmoid":
                return 1-self.sigmoid_function(0, 1, abs(state), 5), False
            elif self.reward_function_tuning == "linear":
                return self.linear_function(1, -1.4, abs(state)), False
            elif self.reward_function_tuning == "pow":
                return pow(1 - abs(state), 5), False
            else:
                return 1 - abs(state), False


    def sigmoid_function(self, start, end, x, slope=10):
        slope = slope / (end - start)
        sigmoid = 1 / (1 + np.exp(-slope * (x - ((start + end) / 2))))
        return sigmoid

    def linear_function(self, cross_x, slope, x):
        return cross_x + (slope * x)

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
