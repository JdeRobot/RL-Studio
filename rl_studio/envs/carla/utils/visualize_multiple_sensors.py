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
        for s in self.sensor_list:
            s.destroy()

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
    ):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

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

        elif sensor_type == "DepthCamera":
            camera_bp = self.world.get_blueprint_library().find("sensor.camera.depth")
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute("image_size_x", str(disp_size[0]))
            camera_bp.set_attribute("image_size_y", str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            # TODO: cambiar
            camera.listen(self.save_depth_image)

            return camera

        elif sensor_type == "DepthLogarithmicCamera":
            camera_bp = self.world.get_blueprint_library().find("sensor.camera.depth")
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute("image_size_x", str(disp_size[0]))
            camera_bp.set_attribute("image_size_y", str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            # TODO: cambiar
            camera.listen(self.save_depthlogarithmic_image)

            return camera

        elif sensor_type == "DvsCamera":
            camera_bp = self.world.get_blueprint_library().find("sensor.camera.dvs")
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute("image_size_x", str(disp_size[0]))
            camera_bp.set_attribute("image_size_y", str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            # TODO: cambiar
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == "OpticalCamera":
            camera_bp = self.world.get_blueprint_library().find(
                "sensor.camera.optical_flow"
            )
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute("image_size_x", str(disp_size[0]))
            camera_bp.set_attribute("image_size_y", str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            # TODO: cambiar
            camera.listen(self.save_optical_image)

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
            # TODO: cambiar
            camera.listen(self.save_semantic_image)

            return camera

        elif sensor_type == "SemanticCameraSergio":
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
            camera.listen(self.save_semantic_image_sergio)

            return camera

        elif sensor_type == "LiDAR":
            lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
            lidar_bp.set_attribute("range", "100")
            lidar_bp.set_attribute(
                "dropoff_general_rate",
                lidar_bp.get_attribute("dropoff_general_rate").recommended_values[0],
            )
            lidar_bp.set_attribute(
                "dropoff_intensity_limit",
                lidar_bp.get_attribute("dropoff_intensity_limit").recommended_values[0],
            )
            lidar_bp.set_attribute(
                "dropoff_zero_intensity",
                lidar_bp.get_attribute("dropoff_zero_intensity").recommended_values[0],
            )

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_lidar_image)

            return lidar

        elif sensor_type == "SemanticLiDAR":
            lidar_bp = self.world.get_blueprint_library().find(
                "sensor.lidar.ray_cast_semantic"
            )
            lidar_bp.set_attribute("range", "100")

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_semanticlidar_image)

            return lidar

        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find("sensor.other.radar")
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar

        else:
            return None

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

        # i = np.array(image.raw_data)
        # print(i.shape)
        # i2 = i.reshape((400, 500, 4))
        # i3 = i2[:, :, :3]
        # cv2.imshow("", array)
        # cv2.waitKey(1)
        # self.front_camera = i3

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

    
    def save_semantic_image_sergio(self, image):
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
        #print(pts_list)

        #for x, y in pts_list:
        #    image_2[x][y] = (255, 0, 0)

        red_line_mask = np.zeros((400, 500, 3), dtype=np.uint8)

        for x, y in pts_list:
            red_line_mask[x][y] = (255, 0, 0)


        if self.display_man.render_enabled():
            #self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self.surface = pygame.surfarray.make_surface(red_line_mask.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1
    

    def save_depth_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Depth)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

    def save_depthlogarithmic_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

    def save_optical_image(self, image):
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

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0 * float(self.sensor_options["range"])

        points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

    def save_semanticlidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0 * float(self.sensor_options["range"])

        points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype("f4"))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += t_end - t_start
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()
