"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass
import time
from datetime import datetime, timedelta

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self, world):
        self.client = None
        self.world = world
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True

        self.actor_list = []

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(VIEW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(VIEW_HEIGHT))
        camera_bp.set_attribute("fov", str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(car_bp, location)
        while self.car is None:
            print(f"entro here {datetime.now()}")
            self.car = self.world.try_spawn_actor(car_bp, location)
        self.actor_list.append(self.car)
        self.car.set_autopilot(True)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)
        )
        self.camera = self.world.spawn_actor(
            self.camera_blueprint(), camera_transform, attach_to=self.car
        )
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (
            2.0 * np.tan(VIEW_FOV * np.pi / 360.0)
        )
        self.camera.calibration = calibration

        self.actor_list.append(self.camera)

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1.0, min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1.0, max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def destroy_all_actors(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
