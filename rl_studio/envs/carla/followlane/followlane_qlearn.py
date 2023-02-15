import math
import time
import carla
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
import numpy as np
import random
from datetime import datetime, timedelta
import weakref
import rospy
from sensor_msgs.msg import Image

from rl_studio.agents.utils import (
    print_messages,
)
from rl_studio.envs.carla.followlane.followlane_env import FollowLaneEnv
from rl_studio.envs.carla.followlane.settings import FollowLaneCarlaConfig

from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
)


class FollowLaneQlearnStaticWeatherNoTraffic(FollowLaneEnv):
    def __init__(self, **config):

        print(f"in FollowLaneQlearnStaticWeatherNoTraffic -> launching FollowLaneEnv\n")
        ###### init F1env
        FollowLaneEnv.__init__(self, **config)
        ###### init class variables
        print(f"leaving FollowLaneEnv\n ")
        print(f"launching FollowLaneCarlaConfig\n ")
        FollowLaneCarlaConfig.__init__(self, **config)

        print(f"config = {config}")
        # ----------------------------
        # self.bsc = config["bsc"]
        # self.world = config["world"]
        # self.camera_rgb_front = config["camera_rgb_front"]
        # self.display_manager = config["display_manager"]

        self.client = carla.Client(
            config["carla_server"],
            config["carla_client"],
        )
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # set syncronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.camera = None
        self.car = None
        self.display = None
        self.image = None

        self.display_manager = DisplayManager(
            grid_size=[2, 3],
            window_size=[1500, 800],
        )

    def reset(self):
        """
        reset for
        - Algorithm: Q-learn
        - State: Simplified perception
        - tasks: FollowLane
        """

        print(f"\nin reset()\n")
        # if len(self.bsc.actor_list) > 0:
        #    print(f"destruyendo actors_list[]")
        #    for actor in self.bsc.actor_list:
        #        actor.destroy()
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.actor_list]
        )
        self.collision_hist = []
        self.actor_list = []
        time.sleep(0.5)

        ## ----  COCHE
        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(car_bp, location)
        while self.car is None:
            # print(f"entro here {datetime.now()}")
            self.car = self.world.try_spawn_actor(car_bp, location)
        self.actor_list.append(self.car)

        print(f"boy por aca")
        ## --- CAMERA
        self.rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"640")
        self.rgb_cam.set_attribute("image_size_y", f"480")
        self.rgb_cam.set_attribute("fov", f"110")
        transform = carla.Transform(
            carla.Location(x=2.5, z=0.7), carla.Rotation(yaw=+00)
        )
        self.front_camera = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.car
        )
        self.actor_list.append(self.front_camera)

        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.front_camera.listen(
            lambda data: FollowLaneQlearnStaticWeatherNoTraffic.process_img(
                weak_self, data
            )
        )

        while self.front_camera is None:
            time.sleep(0.01)

        for actor in self.actor_list:
            print(f"in reset - actor: {actor} \n")

        self.car.set_autopilot(True)

        # self.display_manager.add_sensor(self.front_camera)
        # self.display_manager.render()
        SensorManager(
            self.world,
            self.display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
            self.car,
            {},
            display_pos=[0, 1],
        )
        return self.front_camera

    ####################################################
    ####################################################

    @staticmethod
    def process_img(weak_self, image):
        self = weak_self
        if not self:
            return

        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((480, 640, 4))
        i3 = i2[:, :, :3]
        cv2.imshow("", i3)
        cv2.waitKey(1)
        self.front_camera = i3

    """
    def reset(self):
        from rl_studio.envs.carla.followlane.followlane_env import (
            FollowLaneEnv,
        )

        return FollowLaneEnv.reset(self)

    def step(self, action, step):
        from rl_studio.envs.carla.followlane.followlane_env import (
            FollowLaneEnv,
        )

        return FollowLaneEnv.step(self, action, step)

    """
