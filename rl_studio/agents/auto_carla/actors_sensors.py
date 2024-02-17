import weakref

import carla
import cv2
import numpy as np
import torch


class NewCar(object):
    def __init__(self, parent_actor, init_positions, init=None):
        self.car = None
        self._parent = parent_actor
        self.world = self._parent
        vehicle = self.world.get_blueprint_library().filter("vehicle.*")[0]

        if init is None:
            pose_init = np.random.randint(0, high=len(init_positions))
        else:
            pose_init = init

        location = carla.Transform(
            carla.Location(
                x=init_positions[pose_init][0],
                y=init_positions[pose_init][1],
                z=init_positions[pose_init][2],
            ),
            carla.Rotation(
                pitch=init_positions[pose_init][3],
                yaw=init_positions[pose_init][4],
                roll=init_positions[pose_init][5],
            ),
        )

        self.car = self.world.spawn_actor(vehicle, location)
        while self.car is None:
            self.car = self.world.spawn_actor(vehicle, location)


# ==============================================================================
# -- class RGB CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraRGBSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.front_rgb_camera = None

        self.world = self._parent.get_world()
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", f"640")
        bp.set_attribute("image_size_y", f"480")
        bp.set_attribute("fov", f"110")

        self.sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            attach_to=self._parent,
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraRGBSensor._rgb_image(weak_self, image))

    @staticmethod
    def _rgb_image(weak_self, image):
        """weakref"""
        self = weak_self()
        if not self:
            return
        image = np.array(image.raw_data)
        image = image.reshape((480, 640, 4))
        # image = image.reshape((512, 1024, 4))
        image = image[:, :, :3]
        # self._data_dict["image"] = image3
        self.front_rgb_camera = image


# ==============================================================================
# -- Red Mask CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraRedMaskSemanticSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.front_red_mask_camera = None

        self.world = self._parent.get_world()
        bp = self.world.get_blueprint_library().find(
            "sensor.camera.semantic_segmentation"
        )
        bp.set_attribute("image_size_x", f"640")
        bp.set_attribute("image_size_y", f"480")
        bp.set_attribute("fov", f"110")

        self.sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2, z=1.5), carla.Rotation(yaw=+00)),
            attach_to=self._parent,
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraRedMaskSemanticSensor._red_mask_semantic_image_callback(
                weak_self, image
            )
        )

    @staticmethod
    def _red_mask_semantic_image_callback(weak_self, image):
        """weakref"""
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        hsv_nemo = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)

        if (
            self.world.get_map().name == "Carla/Maps/Town07"
            or self.world.get_map().name == "Carla/Maps/Town04"
            or self.world.get_map().name == "Carla/Maps/Town07_Opt"
            or self.world.get_map().name == "Carla/Maps/Town04_Opt"
        ):
            light_sidewalk = (42, 200, 233)
            dark_sidewalk = (44, 202, 235)
        else:
            light_sidewalk = (151, 217, 243)
            dark_sidewalk = (153, 219, 245)

        light_pavement = (149, 127, 127)
        dark_pavement = (151, 129, 129)

        mask_sidewalk = cv2.inRange(hsv_nemo, light_sidewalk, dark_sidewalk)
        # result_sidewalk = cv2.bitwise_and(array, array, mask=mask_sidewalk)

        mask_pavement = cv2.inRange(hsv_nemo, light_pavement, dark_pavement)
        # result_pavement = cv2.bitwise_and(array, array, mask=mask_pavement)

        # Adjust according to your adjacency requirement.
        kernel = np.ones((3, 3), dtype=np.uint8)

        # Dilating masks to expand boundary.
        color1_mask = cv2.dilate(mask_sidewalk, kernel, iterations=1)
        color2_mask = cv2.dilate(mask_pavement, kernel, iterations=1)

        # Required points now will have both color's mask val as 255.
        common = cv2.bitwise_and(color1_mask, color2_mask)
        SOME_THRESHOLD = 0

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

        # t_end = self.timer.time()
        # self.time_processing += t_end - t_start
        # self.tics_processing += 1

        red_line_mask = cv2.cvtColor(red_line_mask, cv2.COLOR_BGR2RGB)
        self.front_red_mask_camera = red_line_mask

        # AutoCarlaUtils.show_image(
        #    "states",
        #    self.front_red_mask_camera,
        #    600,
        #    400,
        # )
        # if self.front_red_mask_camera is not None:
        #    time.sleep(0.01)
        #    print(f"self.front_red_mask_camera leyendo")
        #    print(f"in _red_mask_semantic_image_callback() {time.time()}")


# ==============================================================================
# -- class LaneDetector -------------------------------------------------------------
# ==============================================================================
class LaneDetector:
    def __init__(self, model_path: str):
        torch.cuda.empty_cache()
        self.__model: torch.nn.Module = torch.load(model_path)
        self.__model.eval()
        self.__threshold = 0.1
        self.__kernel = np.ones((7, 7), np.uint8)
        self.__iterations = 4

    def detect(self, img_array: np.array) -> tuple:
        with torch.no_grad():
            image_tensor = img_array.transpose(2, 0, 1).astype("float32") / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            back, left, right = (
                torch.softmax(self.__model.forward(x_tensor), dim=1).cpu().numpy()[0]
            )

        res, left_mask, right_mask = self._lane_detection_overlay(
            img_array, left, right
        )

        return res, left_mask, right_mask

    def _lane_detection_overlay(
        self, image: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray
    ) -> tuple:
        """
        @type left_mask: object
        """
        res = np.copy(image)

        cv2.erode(left_mask, self.__kernel, self.__iterations)
        cv2.dilate(left_mask, self.__kernel, self.__iterations)

        cv2.erode(left_mask, self.__kernel, self.__iterations)
        cv2.dilate(left_mask, self.__kernel, self.__iterations)

        left_mask = self._image_polyfit(left_mask)
        right_mask = self._image_polyfit(right_mask)

        # We show only points with probability higher than 0.07
        # show lines in BLUE
        # res[left_mask > self.__threshold, :] = [255, 0, 0]  # [255, 0, 0]
        # res[right_mask > self.__threshold, :] = [255, 0, 0]  # [0, 0, 255]

        # show lines in RED
        res[left_mask > self.__threshold, :] = [
            0,
            0,
            255,
        ]  # [0, 0, 255]  # [255, 0, 0]
        res[right_mask > self.__threshold, :] = [
            0,
            0,
            255,
        ]  # [0, 0, 255]  # [0, 0, 255]
        return res, left_mask, right_mask

    def _image_polyfit(self, image: np.ndarray) -> np.ndarray:
        img = np.copy(image)
        img[image > self.__threshold] = 255

        indices = np.where(img == 255)

        if len(indices[0]) == 0:
            return img
        grade = 1
        coefficients = np.polyfit(indices[0], indices[1], grade)

        x = np.linspace(0, img.shape[1], num=2500)
        y = np.polyval(coefficients, x)
        points = np.column_stack((x, y)).astype(int)

        valid_points = []

        for point in points:
            # if (0 < point[1] < 1023) and (0 < point[0] < 509):
            if (0 < point[1] < image.shape[1]) and (0 < point[0] < image.shape[0]):
                valid_points.append(point)

        valid_points = np.array(valid_points)
        polyfitted = np.zeros_like(img)
        polyfitted[tuple(valid_points.T)] = 255

        return polyfitted

    def get_prediction(self, img_array):
        """
        original version, not using it
        """
        with torch.no_grad():
            image_tensor = img_array.transpose(2, 0, 1).astype("float32") / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            model_output = (
                torch.softmax(self.__model.forward(x_tensor), dim=1).cpu().numpy()
            )
        return model_output

    def lane_detection_overlay(self, image, left_mask, right_mask):
        """
        original version, not using it
        """
        res = np.copy(image)
        # We show only points with probability higher than 0.5
        res[left_mask > 0.07, :] = [255, 0, 0]
        res[right_mask > 0.07, :] = [0, 0, 255]

        cv2.erode(left_mask, (7, 7), 4)
        cv2.dilate(left_mask, (7, 7), 4)

        return res
