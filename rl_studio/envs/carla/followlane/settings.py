from pydantic import BaseModel

from rl_studio.envs.carla.followlane.images import AutoCarlaImages
from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils
from rl_studio.envs.carla.followlane.rewards import AutoCarlaRewards
from rl_studio.envs.carla.followlane.simplified_perception import (
    AutoCarlaSimplifiedPerception,
)


class FollowLaneCarlaConfig(BaseModel):
    def __init__(self, **config):
        self.simplifiedperception = AutoCarlaSimplifiedPerception()
        self.autocarlarewards = AutoCarlaRewards()
        self.autocarlautils = AutoCarlaUtils()
        self.autocarlaimages = AutoCarlaImages()

        # self.image = ImageF1()
        # self.image = ListenerCamera("/F1ROS/cameraL/image_raw")
        self.image_raw_from_topic = None
        self.image_camera = None
        # self.sensor = config["sensor"]

        # Image
        self.image_resizing = config["image_resizing"] / 100
        self.new_image_size = config["new_image_size"]
        self.raw_image = config["raw_image"]
        self.height = int(config["height_image"] * self.image_resizing)
        self.width = int(config["width_image"] * self.image_resizing)
        self.center_image = int(config["center_image"] * self.image_resizing)
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        # self.telemetry_mask = config["telemetry_mask"]
        self.poi = config["x_row"][0]
        self.image_center = None
        self.right_lane_center_image = config["center_image"] + (
            config["center_image"] // 2
        )
        self.lower_limit = config["lower_limit"]

        # States
        self.state_space = config["states"]
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

        # Pose
        self.alternate_pose = config["alternate_pose"]
        self.waypoints_meters = config["waypoints_meters"]
        self.waypoints_init = config["waypoints_init"]
        self.waypoints_target = config["waypoints_target"]
        self.waypoints_lane_id = config["waypoints_lane_id"]
        self.waypoints_road_id = config["waypoints_road_id"]

        self.beta = config["beta_1"]
        self.punish_ineffective_vel = config["punish_ineffective_vel"]
        self.punish_zig_zag_value = config["punish_zig_zag_value"]
        # self.actor_list = []
        # self.collision_hist = []
