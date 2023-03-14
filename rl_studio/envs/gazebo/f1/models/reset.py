import numpy as np

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.gazebo.f1.models.f1_env import F1Env


class Reset(F1Env):
    """
    Works for Follow Line and Follow Lane tasks
    """

    def reset_f1_state_image(self):
        """
        reset for
        - State: Image
        - tasks: FollowLane and FollowLine
        """
        self._gazebo_reset()
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_follow_rigth_lane()
        else:
            self._gazebo_set_fix_pose_f1_follow_right_lane()

        self._gazebo_unpause()

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== calculating State
        # image as observation
        state = np.array(
            self.f1gazeboimages.image_preprocessing_black_white_32x32(
                f1_image_camera.data, self.height
            )
        )
        state_size = state.shape

        return state, state_size

    def reset_f1_state_sp(self):
        """
        reset for
        - State: Simplified perception
        - tasks: FollowLane and FollowLine
        """
        self._gazebo_reset()
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_follow_rigth_lane()
        else:
            self._gazebo_set_fix_pose_f1_follow_right_lane()

        self._gazebo_unpause()

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        self._gazebo_pause()

        ##==== calculating State
        # simplified perception as observation
        points_in_red_line, centrals_normalized = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )
        self._gazebo_pause()
        states = centrals_normalized
        state_size = len(states)

        return states, state_size

    def reset_f1_state_sp_line(self):
        """
               reset for
               - State: Simplified perception
               - tasks: FollowLane and FollowLine
               """
        self._gazebo_reset()
        # === POSE ===
        if self.alternate_pose:
            self._gazebo_set_random_pose_f1_follow_rigth_lane()
        else:
            self._gazebo_set_fix_pose_f1_follow_right_lane()

        ##==== get image from sensor camera
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()
        f1_image_camera, _ = self.f1gazeboimages.get_camera_info()

        ##==== get center
        points_in_red_line, _ = self.simplifiedperception.processed_image(
            f1_image_camera.data, self.height, self.width, self.x_row, self.center_image
        )

        if self.state_space == "spn":
            self.point = points_in_red_line[self.poi]
        else:
            self.point = points_in_red_line[0]

        ##==== get State
        ##==== simplified perception as observation
        states = self.simplifiedperception.calculate_observation(
            points_in_red_line, self.center_image, self.pixel_region
        )

        state_size = len(states)
        
        return states, state_size
