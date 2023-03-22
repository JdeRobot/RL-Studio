import cv2
import imutils
import numpy as np


class AutoCarlaUtils:
    def __init__(self):
        self.f1 = None

    @staticmethod
    def finish_target(current_car_pose, target_pose, max_distance):

        current_car_pose_x = current_car_pose[0]
        current_car_pose_y = current_car_pose[1]
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
    def show_image_with_everything(
        name, img, waitkey, centrals_in_pixels, errors, states, x_row, x, y
    ):
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
            ### Points
            cv2.circle(
                img,
                (int(centrals_in_pixels[index]), int(x_row[index])),
                5,
                # (150, 200, 150),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                img,
                str(
                    f"[center:{int(centrals_in_pixels[index])}]-[state:{states[index]}]-[dist:{errors[index]}]"
                ),
                (int(centrals_in_pixels[index]) - 50, int(x_row[index]) - 5),
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
