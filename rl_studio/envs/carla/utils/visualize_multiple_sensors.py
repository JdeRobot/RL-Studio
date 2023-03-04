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

import carla
import argparse
import random
import time
import numpy as np
import cv2

from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class DisplayManager:
    def __init__(self, grid_size, window_size):

        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []
        # self.actor_list = []
        # self.visualize = visualize

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [
            int(self.window_size[0] / self.grid_size[1]),
            int(self.window_size[1] / self.grid_size[0]),
        ]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        # print(f"entro en destroy()")
        for s in self.sensor_list:
            s.destroy()
        self.sensor_list = []

    def render_enabled(self):
        return self.display != None


class SensorManager:
    def __init__(
        self,
        world,
        display_man,
        sensor_type,
        transform,
        attached,
        sensor_options,
        display_pos,
        client=None,
    ):
        self.surface = None
        self.world = world
        self.client = client  # added for BEV
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()
        self.time_processing = 0.0
        self.tics_processing = 0
        self.display_man.add_sensor(self)

        self.front_camera = None
        self.front_camera_red_mask = None
        self.front_camera_bev = None

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == "RGBCamera":
            camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute("image_size_x", str(disp_size[0]))
            camera_bp.set_attribute("image_size_y", str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == "SemanticCamera":
            camera_bp = self.world.get_blueprint_library().find(
                "sensor.camera.semantic_segmentation"
            )
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute("image_size_x", str(disp_size[0]))
            camera_bp.set_attribute("image_size_y", str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_semantic_image)

            return camera

        elif sensor_type == "RedMask":
            camera_bp = self.world.get_blueprint_library().find(
                "sensor.camera.semantic_segmentation"
            )
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute("image_size_x", str(disp_size[0]))
            camera_bp.set_attribute("image_size_y", str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            # TODO: cambiar
            camera.listen(self.save_red_mask_semantic_image)

            return camera

        elif sensor_type == "BirdEyeView":
            # client = carla.Client("localhost", 2000)
            # client.set_timeout(10.0)

            self.birdview_producer = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=100, height=300),
                pixels_per_meter=10,
                crop_type=BirdViewCropType.FRONT_AREA_ONLY,
            )

            camera_bp = self.world.get_blueprint_library().find(
                "sensor.camera.semantic_segmentation"
            )
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute("image_size_x", str(disp_size[0]))
            camera_bp.set_attribute("image_size_y", str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_bev_image)

            return camera

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

        self.front_camera = array

    def save_semantic_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

    def save_red_mask_semantic_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        hsv_nemo = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        light_sidewalk = (151, 217, 243)
        dark_sidewalk = (153, 219, 245)

        light_pavement = (149, 127, 127)
        dark_pavement = (151, 129, 129)

        mask_sidewalk = cv2.inRange(hsv_nemo, light_sidewalk, dark_sidewalk)
        result_sidewalk = cv2.bitwise_and(array, array, mask=mask_sidewalk)

        mask_pavement = cv2.inRange(hsv_nemo, light_pavement, dark_pavement)
        result_pavement = cv2.bitwise_and(array, array, mask=mask_pavement)

        # Adjust according to your adjacency requirement.
        kernel = np.ones((3, 3), dtype=np.uint8)

        # Dilating masks to expand boundary.
        color1_mask = cv2.dilate(mask_sidewalk, kernel, iterations=1)
        color2_mask = cv2.dilate(mask_pavement, kernel, iterations=1)

        # Required points now will have both color's mask val as 255.
        common = cv2.bitwise_and(color1_mask, color2_mask)
        SOME_THRESHOLD = 10

        # Common is binary np.uint8 image, min = 0, max = 255.
        # SOME_THRESHOLD can be anything within the above range. (not needed though)
        # Extract/Use it in whatever way you want it.
        intersection_points = np.where(common > SOME_THRESHOLD)

        # Say you want these points in a list form, then you can do this.
        pts_list = [[r, c] for r, c in zip(*intersection_points)]
        # print(pts_list)

        # for x, y in pts_list:
        #    image_2[x][y] = (255, 0, 0)

        # red_line_mask = np.zeros((400, 500, 3), dtype=np.uint8)
        red_line_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

        for x, y in pts_list:
            red_line_mask[x][y] = (255, 0, 0)

        if self.display_man.render_enabled():
            # self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self.surface = pygame.surfarray.make_surface(red_line_mask.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

        self.front_camera_red_mask = red_line_mask

    def save_bev_image(self, image):
        t_start = self.timer.time()
        if self.display_man.render_enabled():
            car_bp = self.world.get_actors().filter("vehicle.*")[0]
            birdview = self.birdview_producer.produce(
                agent_vehicle=car_bp  # carla.Actor (spawned vehicle)
            )
            image = BirdViewProducer.as_rgb(birdview)
            image = np.rot90(image)

            if image.shape[0] != image.shape[1]:
                if image.shape[0] > image.shape[1]:
                    difference = image.shape[0] - image.shape[1]
                    extra_left, extra_right = int(difference / 2), int(difference / 2)
                    extra_top, extra_bottom = 0, 0
                else:
                    difference = image.shape[1] - image.shape[0]
                    extra_left, extra_right = 0, 0
                    extra_top, extra_bottom = int(difference / 2), int(difference / 2)
                image = np.pad(
                    image,
                    ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                image = np.pad(
                    image,
                    ((100, 100), (50, 50), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            self.surface = pygame.surfarray.make_surface(image)

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

        self.front_camera_bev = image

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        # print(f"in DisplayManeger.destroy - self.sensor = {self.sensor}")
        self.sensor.destroy()
