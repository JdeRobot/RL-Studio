import cv2
import numpy as np

from rl_studio.agents.f1.settings import QLearnConfig


class ImageF1:

    font = cv2.FONT_HERSHEY_COMPLEX

    def __init__(self):
        self.height = 3  # Image height [pixels]
        self.width = 3  # Image width [pixels]
        self.timeStamp = 0  # Time stamp [s] */
        self.format = ""  # Image format string (RGB8, BGR,...)
        self.data = np.zeros(
            (self.height, self.width, 3), np.uint8
        )  # The image data itself
        self.data.shape = self.height, self.width, 3
        self.config = QLearnConfig()

    def __str__(self):
        return (
            f"Image:"
            f"\nHeight: {self.height}\nWidth: {self.width}\n"
            f"Format: {self.format}\nTimeStamp: {self.timeStamp}\nData: {self.data}"
        )

    @staticmethod
    def image_msg_to_image(img, cv_image):

        img.width = img.width
        img.height = img.height
        img.format = "RGB8"
        img.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        img.data = cv_image

        return img

    def show_telemetry(self, img, points, action, reward):
        count = 0
        for idx, point in enumerate(points):
            cv2.line(
                img,
                (320, self.config.x_row[idx]),
                (320, self.config.x_row[idx]),
                (255, 255, 0),
                thickness=5,
            )
            # cv2.line(img, (center_image, x_row[idx]), (point, x_row[idx]), (255, 255, 255), thickness=2)
            cv2.putText(
                img,
                str("err{}: {}".format(idx + 1, self.config.center_image - point)),
                (18, 340 + count),
                self.font,
                0.4,
                (255, 255, 255),
                1,
            )
            count += 20
        cv2.putText(
            img,
            str(f"action: {action}"),
            (18, 280),
            self,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            img,
            str(f"reward: {reward}"),
            (18, 320),
            self,
            0.4,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Image window", img[240:])
        cv2.waitKey(3)
