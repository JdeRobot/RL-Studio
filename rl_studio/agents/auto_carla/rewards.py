import math

import numpy as np


class AutoCarlaRewards:

    def rewards_followlane_center_velocity_angle(
        self, centers_norm, velocity, target_vel, angle
    ):
        """
        rewards = f(center, vel, angle) = center + velocity - heading

        """

        w_center = 1
        w_velocity = 1
        w_heading = 1
        done_function = False

        center_reward, done_center, centers_rewards_list = self.rewards_center(
            centers_norm
        )
        velocity_reward, done_velocity = self.rewards_velocity(velocity, target_vel)
        heading_reward, done_heading = self.rewards_heading(angle)

        # print(f"\n\tin rewards function()....")
        # print(f"\t{center_reward=}, {done_center=}, {centers_rewards_list=}")
        # print(f"\t{velocity_reward=}, {done_velocity=}")
        # print(f"\t{heading_reward=}, {done_heading=}")

        function_reward = (
            (w_center * center_reward)
            + (w_velocity * velocity_reward)
            + (w_heading * heading_reward)
        )
        if done_center or done_velocity:  # or done_heading:
            done_function = True

        return (
            center_reward,
            done_center,
            centers_rewards_list,
            velocity_reward,
            done_velocity,
            heading_reward,
            done_heading,
            done_function,
            function_reward,
        )
        # return function_reward, done_function, centers_rewards_list

    def rewards_center(self, centers_norm):
        """
        Sigmoid
        rewards in x = [0, 1] and y = [0, 1]
        a - (b / (c + d * exp(-i * center_norm)))
        """

        a = 1.078
        b = 0.108
        c = 0.1
        d = 1.3
        i = 8.7

        rewards = []
        done = False
        # for index, _ in enumerate(centers_norm):
        #    dist = dist_normalized[index] - ground_truth_normalized[index]
        #  rewards.append(a - (b / (c + d * math.exp(-i * centers_norm[index]))))
        rewards = [
            a - (b / (c + d * math.exp(-i * abs(centers_norm[index]))))
            for index, _ in enumerate(centers_norm)
        ]

        function_reward = sum(rewards) / len(rewards)
        # dist_normaliz_mean = sum(dist_normalized) / len(dist_normalized)

        # if function_reward < 0.1: #por dist_normaliz_mean >= 0.16 or dist_normaliz_mean <= -0.5):  # distance of -0.8 to right, and 0.6 to left
        # if max([abs(value) for value in centers_norm]) > max_dist2center_allowed: #por dist_normaliz_mean >= 0.16 or dist_normaliz_mean <= -0.5):  # distance of -0.8 to right, and 0.6 to left
        if (
            function_reward < 0.2
        ):  # por dist_normaliz_mean >= 0.16 or dist_normaliz_mean <= -0.5):  # distance of -0.8 to right, and 0.6 to left
            done = True
            done = True
        #  function_reward = 0

        return function_reward, done, rewards

    def rewards_velocity(self, velocity, target_vel=50):
        """
        From vel = [0, target_vel] is a increasing linear function from [0,1]
        From vel > target_vel, the function is a sharply decreasing sigmoid
        """
        done = False
        # x_values = np.linspace(0, max_veloc, 1000)

        if (
            velocity <= target_vel
        ):  # up to target_vel the function is a simple increasing linear function
            reward = (velocity - 0) / (target_vel - 0)
        else:  # after target_vel ( ie. > 50 the function is decreasing sharply sigmoid )
            # return 1.0 - (x - 50) / (200 - 50)
            # return 1 / (1 + np.exp((x - 95) / 14.4))
            reward = 1.0 - (1 / (1 + np.exp(-0.5 * (velocity - target_vel))))

        # if reward < 0.1:
        if not (0 <= velocity <= target_vel + 5):
            done = True

        return reward, done

    def rewards_heading(self, angle, max_angle=30):
        """
        rewards in x = [0, 1] and y = [0, 1]
        It is a sigmoid function
        """

        a = 1
        b = 1
        c = 1
        d = -12.4
        i = 0.4

        # rewards = []
        done = False
        angle_normal = angle / max_angle

        reward = a - (b / (c + math.exp(d * (abs(angle_normal) - i))))

        if abs(angle_normal) >= 0.9:
            done = True

        return reward, done

    ######################################

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
