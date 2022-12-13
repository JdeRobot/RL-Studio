import math

import numpy as np


def calculate_reward(self, error: float) -> float:
    d = np.true_divide(error, self.center_image)
    reward = np.round(np.exp(-d), 4)
    return reward


def rewards_discrete_follow_lane(self, center):
    """
    works perfectly
    """
    done = False
    if 0.65 >= center > 0.25:
        reward = self.rewards["from_10"]
    elif (0.9 > center > 0.65) or (0.25 >= center > 0):
        reward = self.rewards["from_02"]
    elif 0 >= center > -0.9:
        reward = self.rewards["from_01"]
    else:
        reward = self.rewards["penal"]
        done = True

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


def reward_v_center_step(self, vel_cmd, center, step):

    done = False
    if 0.65 >= center > 0.25:
        reward = (self.rewards["from_10"] + vel_cmd.linear.x) - math.log(step)
    elif (0.9 > center > 0.65) or (0.25 >= center > 0):
        reward = (self.rewards["from_02"] + vel_cmd.linear.x) - math.log(step)
    elif 0 >= center > -0.9:
        # reward = (self.rewards["from_01"] + vel_cmd.linear.x) - math.log(step)
        reward = -math.log(step)
    else:
        reward = self.rewards["penal"]
        done = True

    return reward, done


def rewards_discrete_follow_right_lane(self, centrals_in_pixels, centrals_normalized):
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


def reward_v_w_center_linear_no_working_at_all(self, vel_cmd, center):
    # print_messages(
    #    "in reward_v_w_center_linear()",
    #    beta1=self.beta_1,
    #    beta0=self.beta_0,
    # )

    w_target = self.beta_0 - (self.beta_1 * abs(vel_cmd.linear.x))
    w_error = abs(w_target - abs(vel_cmd.angular.z))
    done = False

    if abs(center) > 0.9 or center < 0:
        done = True
        reward = self.rewards["penal"]
    elif center >= 0:
        reward = (
            (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
        )  # add a constant to favor right lane
        # else:
        #    reward = (1 / math.exp(w_error)) + (math.exp(center))

    return reward, done


def reward_v_w_center_linear_second(self, vel_cmd, center):
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

    w_target = self.beta_0 - (self.beta_1 * abs(vel_cmd.linear.x))
    w_error = abs(w_target - abs(vel_cmd.angular.z))
    done = False

    if abs(center) > 0.9:
        done = True
        reward = self.rewards["penal"]
    elif center > 0:
        reward = (
            (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
        )  # add a constant to favor right lane
    else:
        reward = (1 / math.exp(w_error)) + (math.exp(center))

    return reward, done


def reward_v_w_center_linear_first_formula(self, vel_cmd, center):
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

    return round(reward, 3)

    #################################################################################
    # Rewards
    #################################################################################


'''    
    def calculate_reward(self, error: float) -> float:
        d = np.true_divide(error, self.center_image)
        reward = np.round(np.exp(-d), 4)
        return reward

    def rewards_discrete_follow_lane(self, center):
        """
        works perfectly
        """
        done = False
        if 0.65 >= center > 0.25:
            reward = self.rewards["from_10"]
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = self.rewards["from_02"]
        elif 0 >= center > -0.9:
            reward = self.rewards["from_01"]
        else:
            reward = self.rewards["penal"]
            done = True

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

    def reward_v_center_step(self, vel_cmd, center, step):

        done = False
        if 0.65 >= center > 0.25:
            reward = (self.rewards["from_10"] + vel_cmd.linear.x) - math.log(step)
        elif (0.9 > center > 0.65) or (0.25 >= center > 0):
            reward = (self.rewards["from_02"] + vel_cmd.linear.x) - math.log(step)
        elif 0 >= center > -0.9:
            # reward = (self.rewards["from_01"] + vel_cmd.linear.x) - math.log(step)
            reward = -math.log(step)
        else:
            reward = self.rewards["penal"]
            done = True

        # if abs(center) > 0.9:  # out for both sides: left and right
        # done = True
        #    reward = -100
        # elif 0 > center:  # in left lane
        #    reward = 0
        # else:  # in right lane
        #    try:
        #        reward = ((0.1 * vel_cmd.linear.x) / center) - math.log(step)
        #    except:
        #        reward = ((0.1 * vel_cmd.linear.x) / 0.1) - math.log(step)

        return reward, done

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

    def reward_v_w_center_linear_no_working_at_all(self, vel_cmd, center):
        # print_messages(
        #    "in reward_v_w_center_linear()",
        #    beta1=self.beta_1,
        #    beta0=self.beta_0,
        # )

        w_target = self.beta_0 - (self.beta_1 * abs(vel_cmd.linear.x))
        w_error = abs(w_target - abs(vel_cmd.angular.z))
        done = False

        if abs(center) > 0.9 or center < 0:
            done = True
            reward = self.rewards["penal"]
        elif center >= 0:
            reward = (
                (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
            )  # add a constant to favor right lane
        # else:
        #    reward = (1 / math.exp(w_error)) + (math.exp(center))

        return reward, done

    def reward_v_w_center_linear_second(self, vel_cmd, center):
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

        w_target = self.beta_0 - (self.beta_1 * abs(vel_cmd.linear.x))
        w_error = abs(w_target - abs(vel_cmd.angular.z))
        done = False

        if abs(center) > 0.9:
            done = True
            reward = self.rewards["penal"]
        elif center > 0:
            reward = (
                (1 / math.exp(w_error)) + (1 / math.exp(center)) + 2
            )  # add a constant to favor right lane
        else:
            reward = (1 / math.exp(w_error)) + (math.exp(center))

        return reward, done

    def reward_v_w_center_linear_first_formula(self, vel_cmd, center):
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

        return round(reward, 3)
'''
