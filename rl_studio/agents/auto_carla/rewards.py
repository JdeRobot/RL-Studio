import math

import numpy as np


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
