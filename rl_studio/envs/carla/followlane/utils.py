import cv2
import numpy as np


class AutoCarlaUtils:
    def __init__(self):
        self.f1 = None

    @staticmethod
    def show_image_with_centrals(
        name, img, waitkey, centrals_in_pixels, centrals_normalized, x_row
    ):
        window_name = f"{name}"
        img = np.array(img)

        for index, _ in enumerate(x_row):
            cv2.putText(
                img,
                str(f"{int(centrals_in_pixels[index])}"),
                (int(centrals_in_pixels[index]) + 20, int(x_row[index])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                str(f"[{centrals_normalized[index]}]"),
                (320, int(x_row[index])),
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
                color=(10, 10, 10),
                thickness=1,
            )
            cv2.line(
                img,
                (int(img.shape[1] // 2), 0),
                (int(img.shape[1] // 2), int(img.shape[0])),
                color=(30, 30, 30),
                thickness=1,
            )

        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    @staticmethod
    def show_image(name, img, waitkey):
        window_name = f"{name}"
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)
