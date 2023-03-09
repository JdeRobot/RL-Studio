import cv2
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
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                str(f"[{centrals_normalized[index]}]"),
                (img.shape[1] - 50, int(x_row[index])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                img,
                (0, int(x_row[index])),
                (int(img.shape[1]), int(x_row[index])),
                color=(200, 200, 200),
                thickness=1,
            )
            cv2.line(
                img,
                (int(img.shape[1] // 2), 0),
                (int(img.shape[1] // 2), int(img.shape[0])),
                color=(200, 200, 200),
                thickness=1,
            )

        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    @staticmethod
    def show_image(name, img, waitkey, x, y):
        window_name = f"{name}"
        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, x, y)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)
