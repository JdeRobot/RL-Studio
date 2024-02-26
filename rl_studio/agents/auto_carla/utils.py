import logging
import imutils
import sys

import cv2
import numpy as np


def format_time(seconds):
    """
    Convert the elapsed time from seconds into a human-readable format.
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = seconds * 1000
    seconds, milliseconds = divmod(milliseconds, 1000)
    return "{:02}:{:02}:{:02}.{:03}".format(
        int(hours), int(minutes), int(seconds), int(milliseconds)
    )


def get_variables_size():
    total_size = 0
    all_vars = {
        k: v for k, v in globals().items() if not k.startswith("_")
    }  # Exclude variables starting with _
    all_vars.update(locals())  # Include local variables
    for var_name, var_value in all_vars.items():
        if not var_name.startswith("_"):
            size = sys.getsizeof(var_value)
            # print(f"Size of {var_name}: {size} bytes")
            total_size += size

    return all_vars, total_size


def old_get_variables_size():
    """
    Get the size of declared variables in global and local scopes

    using it:

    # Get the dictionary of variable names and sizes
    variables_size = get_variables_size()

    # Print the dictionary
    for var_name, size in variables_size.items():
        print(f"{var_name}: {size} bytes")

    """
    all_vars = {}
    all_vars.update(globals())  # Add global variables
    all_vars.update(locals())  # Add local variables

    variables_size = {}
    for var_name, var in all_vars.items():
        # Filter out built-in and module variables
        if (
            not var_name.startswith("__")
            and not hasattr(var, "__module__")
            and not var_name.startswith("_")
        ):
            variables_size[var_name] = sys.getsizeof(var)
    return variables_size


class Logger:
    def __init__(self, log_file):
        # Configurar el formato de los registros
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.DEBUG, format=format)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(
            logging.WARNING
        )  # Solo se escribirán advertencias y errores en el archivo
        file_handler.setFormatter(logging.Formatter(format))
        logging.getLogger().addHandler(file_handler)

        # Configurar un manipulador para imprimir en la consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            logging.INFO
        )  # Solo se imprimirán mensajes de info y debug en la consola
        console_handler.setFormatter(logging.Formatter(format))
        logging.getLogger().addHandler(console_handler)

    def _info(self, message):
        logging.info(message)

    def _warning(self, message):
        logging.warning(message)

    def _error(self, message):
        logging.error(message)

    def _debug(self, message):
        logging.debug(message)


class LoggerAllInOne:
    def __init__(self, log_file):
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(filename=log_file, level=logging.DEBUG, format=format)
        self.logger = logging.getLogger(__name__)

    def _info(self, message):
        self.logger.info(message)

    def _warning(self, message):
        self.logger.warning(message)

    def _error(self, message):
        self.logger.error(message)

    def _debug(self, message):
        self.logger.debug(message)


class LoggingHandler:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)

        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter(
            "[%(levelname)s] - %(asctime)s, filename: %(filename)s, funcname: %(funcName)s, line: %(lineno)s\n messages ---->\n %(message)s"
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)


class AutoCarlaUtils:
    def __init__(self):
        self.f1 = None

    @staticmethod
    def finish_target(current_car_pose, target_pose, max_distance):
        """
        working with waypoints
        """
        current_car_pose_x = current_car_pose[0]
        current_car_pose_y = current_car_pose[1]

        # in case working with waypoints, use next
        target_x = target_pose.transform.location.x
        target_y = target_pose.transform.location.y

        dist = ((current_car_pose_x - target_x) ** 2) + (
            (current_car_pose_y - target_y) ** 2
        )
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)

        # print(f"{current_car_pose = }")
        # print(f"{target_x = }, {target_y = }")
        # print(f"{dist = }")

        # print(dist)
        if dist < max_distance:
            return True, dist
        return False, dist

    @staticmethod
    def finish_fix_number_target(
        current_car_pose, targets_set, target_pose, max_distance
    ):
        """
        working with tuple 0: [19.8, -238.2, 0.5, -4.5, -150.5, -0.5]
        """
        current_car_pose_x = current_car_pose[0]
        current_car_pose_y = current_car_pose[1]

        target_x = targets_set[target_pose][0]
        target_y = targets_set[target_pose][1]

        dist = ((current_car_pose_x - target_x) ** 2) + (
            (current_car_pose_y - target_y) ** 2
        )
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)

        # print(f"{current_car_pose = }")
        # print(f"{target_x = }, {target_y = }")
        # print(f"{dist = }")

        # print(dist)
        if dist < max_distance:
            return True, dist
        return False, dist

    @staticmethod
    def show_image_lines_centers_borders(
        name,
        img,
        x_row,
        x,
        y,
        index_right,
        index_left,
        centers,
        drawing_lines_states,
        drawing_numbers_states,
    ):
        """
        show image RGB with:
            x_row lines +
            centers points +
            center image line +
            lane borders points +
            heading vertical line

        """

        window_name = f"{name}"
        img = np.array(img)

        ## vertical line in the center of image, showing car position
        cv2.line(
            img,
            (int(img.shape[1] // 2), int(img.shape[0] // 2)),
            (int(img.shape[1] // 2), int(img.shape[0])),
            # (320, 120),
            # (320, 480),
            color=(200, 100, 100),
            thickness=4,
        )

        ## heading line from upper center point to lower center point
        cv2.line(
            img,
            (centers[0], x_row[0]),
            (centers[-1], x_row[-1]),
            # (320, 120),
            # (320, 480),
            color=(0, 0, 255),
            thickness=2,
        )

        drawing_lines_states.append(640)
        ## vertical lines for states: 5, 7, 8, 16...
        for index, _ in enumerate(drawing_lines_states):
            cv2.line(
                img,
                (drawing_lines_states[index], 0),
                (drawing_lines_states[index], int(img.shape[0])),
                color=(100, 200, 100),
                thickness=1,
            )
            ## writing number state into lines
            cv2.putText(
                img,
                str(f"{drawing_numbers_states[index]}"),
                (
                    drawing_lines_states[index] - 15,
                    15,
                ),  ## -15 is the distance to the its left line
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        for index, _ in enumerate(x_row):
            ### HORIZONTAL LINES x_row
            cv2.line(
                img,
                (0, int(x_row[index])),
                (int(img.shape[1]), int(x_row[index])),
                color=(100, 200, 100),
                thickness=1,
            )

            ### Points
            cv2.circle(
                img,
                (int(index_right[index]), x_row[index]),
                5,
                # (150, 200, 150),
                (0, 0, 255),
                2,
            )
            cv2.circle(
                img,
                (int(index_left[index]), int(x_row[index])),
                4,
                # (150, 200, 150),
                (0, 0, 255),
                1,
            )
            cv2.circle(
                img,
                (int(centers[index]), int(x_row[index])),
                4,
                # (150, 200, 150),
                (0, 0, 0),
                1,
            )

        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    @staticmethod
    def show_image_only_right_line(
        name,
        img,
        waitkey,
        dist_in_pixels,
        ground_truth_pixel_values,
        dist_normalized,
        states,
        x_row,
        x,
        y,
        line_states,
        number_states,
    ):
        """
        shows image with 1 point in center of lane
        """
        window_name = f"{name}"
        img = np.array(img)

        ## vertical line in the center of image, showing car position
        cv2.line(
            img,
            (int(img.shape[1] // 2), int(img.shape[0] // 2)),
            (int(img.shape[1] // 2), int(img.shape[0])),
            # (320, 120),
            # (320, 480),
            color=(200, 100, 100),
            thickness=4,
        )

        line_states.append(640)
        ## vertical lines for states: 5, 7, 8, 16...
        for index, _ in enumerate(line_states):
            cv2.line(
                img,
                (line_states[index], 0),
                (line_states[index], int(img.shape[0])),
                color=(100, 200, 100),
                thickness=1,
            )
            ## writing number state into lines
            # for index, value in enumerate(number_states):
            cv2.putText(
                img,
                str(f"{number_states[index]}"),
                (line_states[index] - 30, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        for index, _ in enumerate(x_row):
            ### Points
            cv2.circle(
                img,
                (
                    int(dist_in_pixels[index]),
                    int(x_row[index]),
                ),
                5,
                # (150, 200, 150),
                (255, 255, 255),
                2,
            )
            cv2.circle(
                img,
                (int(ground_truth_pixel_values[index]), int(x_row[index])),
                4,
                # (150, 200, 150),
                (255, 255, 255),
                1,
            )

            cv2.putText(
                img,
                str(
                    f"[right_line:{int(dist_in_pixels[index])}]-[dist:{dist_normalized[index]}]"
                    # f"[dist_norm:{int(centrals_in_pixels[index])}]-[state:{states[index]}]-[dist:{errors[index]}]"
                ),
                (int(dist_in_pixels[index]), int(x_row[index]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                # (255, 255, 255),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    @staticmethod
    def show_image_with_three_points(
        name,
        img,
        waitkey,
        dist_in_pixels,
        dist_normalized,
        states,
        x_row,
        x,
        y,
        left_points,
        right_points,
    ):
        """
        shows image with 1 point in center of lane
        """
        window_name = f"{name}"
        img = np.array(img)

        for index, _ in enumerate(x_row):
            ### horizontal line in x_row file
            cv2.line(
                img,
                (0, int(x_row[index])),
                (int(img.shape[1]), int(x_row[index])),
                color=(100, 200, 100),
                thickness=1,
            )
            ### vertical line in center of the image
            cv2.line(
                img,
                (int(img.shape[1] // 2), 0),
                (int(img.shape[1] // 2), int(img.shape[0])),
                color=(100, 200, 100),
                thickness=1,
            )
            ### left limit vertical line (40%)
            cv2.line(
                img,
                (int(img.shape[1] // 2) - (int((img.shape[1] // 2) * 0.4)), 0),
                (
                    int(img.shape[1] // 2) - (int((img.shape[1] // 2) * 0.4)),
                    int(img.shape[0]),
                ),
                color=(100, 200, 100),
                thickness=1,
            )
            ### right limit vertical line
            cv2.line(
                img,
                (int(img.shape[1] // 2) + (int((img.shape[1] // 2) * 0.4)), 0),
                (
                    int(img.shape[1] // 2) + (int((img.shape[1] // 2) * 0.4)),
                    int(img.shape[0]),
                ),
                color=(100, 200, 100),
                thickness=1,
            )

            ### Points
            ## center point
            cv2.circle(
                img,
                (int(dist_in_pixels[index]), int(x_row[index])),
                5,
                # (150, 200, 150),
                (255, 255, 255),
                2,
            )
            # left point
            cv2.circle(
                img,
                (int(left_points[index]), int(x_row[index])),
                5,
                # (150, 200, 150),
                (255, 255, 255),
                2,
            )
            # right point
            cv2.circle(
                img,
                (int(right_points[index]), int(x_row[index])),
                5,
                # (150, 200, 150),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                img,
                str(
                    f"[right_line:{int(dist_in_pixels[index])}]-[state:{states[index]}]-[dist:{dist_normalized[index]}]"
                    # f"[dist_norm:{int(centrals_in_pixels[index])}]-[state:{states[index]}]-[dist:{errors[index]}]"
                ),
                (int(dist_in_pixels[index]), int(x_row[index]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                # (255, 255, 255),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    @staticmethod
    def show_image_with_center_point(
        name, img, waitkey, dist_in_pixels, dist_normalized, states, x_row, x, y
    ):
        """
        shows image with 1 point in center of lane
        """
        window_name = f"{name}"
        img = np.array(img)

        for index, _ in enumerate(x_row):
            ### horizontal line in x_row file
            cv2.line(
                img,
                (0, int(x_row[index])),
                (int(img.shape[1]), int(x_row[index])),
                color=(100, 200, 100),
                thickness=1,
            )
            ### vertical line in center of the image
            cv2.line(
                img,
                (int(img.shape[1] // 2), 0),
                (int(img.shape[1] // 2), int(img.shape[0])),
                color=(100, 200, 100),
                thickness=1,
            )
            ### left limit vertical line (40%)
            cv2.line(
                img,
                (int(img.shape[1] // 2) - (int((img.shape[1] // 2) * 0.4)), 0),
                (
                    int(img.shape[1] // 2) - (int((img.shape[1] // 2) * 0.4)),
                    int(img.shape[0]),
                ),
                color=(100, 200, 100),
                thickness=1,
            )
            ### right limit vertical line
            cv2.line(
                img,
                (int(img.shape[1] // 2) + (int((img.shape[1] // 2) * 0.4)), 0),
                (
                    int(img.shape[1] // 2) + (int((img.shape[1] // 2) * 0.4)),
                    int(img.shape[0]),
                ),
                color=(100, 200, 100),
                thickness=1,
            )

            ### Points
            cv2.circle(
                img,
                (
                    int(img.shape[1] // 2) + int(dist_in_pixels[index]),
                    int(x_row[index]),
                ),
                5,
                # (150, 200, 150),
                (255, 255, 255),
                2,
            )
            cv2.circle(
                img,
                (int(img.shape[1] // 2), int(x_row[index])),
                4,
                # (150, 200, 150),
                (255, 255, 255),
                1,
            )

            cv2.putText(
                img,
                str(
                    f"[right_line:{int(dist_in_pixels[index])}]-[state:{states[index]}]-[dist:{dist_normalized[index]}]"
                    # f"[dist_norm:{int(centrals_in_pixels[index])}]-[state:{states[index]}]-[dist:{errors[index]}]"
                ),
                (int(dist_in_pixels[index]), int(x_row[index]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                # (255, 255, 255),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    @staticmethod
    def show_image_with_centrals(
        name, img, waitkey, centrals_in_pixels, centrals_normalized, x_row, x, y
    ):
        window_name = f"{name}"
        img = np.array(img)

        for index, _ in enumerate(x_row):
            cv2.putText(
                img,
                str(f"{int(centrals_in_pixels[index])}"),
                (
                    (img.shape[1] // 2) + int(centrals_in_pixels[index]) + 20,
                    int(x_row[index]),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                # (255, 255, 255),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                str(f"[{centrals_normalized[index]}]"),
                (img.shape[1] - 100, int(x_row[index])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                # (255, 255, 255),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                img,
                (0, int(x_row[index])),
                (int(img.shape[1]), int(x_row[index])),
                color=(100, 200, 100),
                thickness=1,
            )
            cv2.line(
                img,
                (int(img.shape[1] // 2), 0),
                (int(img.shape[1] // 2), int(img.shape[0])),
                color=(100, 200, 100),
                thickness=1,
            )

        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    @staticmethod
    def show_image(name, img, x, y):
        window_name = f"{name}"
        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    @staticmethod
    def show_images(name, img, x, y):
        window_name = f"{name}"
        hori = np.concatenate(img, axis=1)
        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, hori)
        cv2.waitKey(1)

    @staticmethod
    def show_images_tile(name, list_img, x, y):
        window_name = f"{name}"
        # img_tile = cv2.vconcat([cv2.hconcat(list_h) for list_h in list_img])

        width_common = 320
        height_common = 240
        # image resizing
        # im_list_resize = [
        #    cv2.resize(
        #        np.array(img),
        #        (width_common, height_common),
        #        interpolation=cv2.INTER_CUBIC,
        #    )
        #    for img in list_img
        # ]
        im_list_resize = [
            imutils.resize(
                np.array(img),
                width=width_common,
            )
            for img in list_img
        ]
        # print(f"{im_list_resize = }")
        # for i, value in enumerate(im_list_resize):
        #    print(f"({i = }")
        # print(f"{len(im_list_resize) = }, {type(im_list_resize) = }")
        im_list = [img for img in im_list_resize]
        # print(f"{len(im_list) = }, {type(im_list) = }")

        im_list_concat = np.concatenate(im_list, axis=1)
        # im_list_concat = cv2.hconcat(
        #    [im_list_resize[0], im_list_resize[1], im_list_resize[2]]
        # )
        # im_list_concat = cv2.hconcat([im_list_resize[0], im_list_resize[1]])
        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, im_list_concat)
        cv2.waitKey(1)
