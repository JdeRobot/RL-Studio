import cv2
import numpy as np


class ImageF1:

    font = cv2.FONT_HERSHEY_COMPLEX

    def __init__(self):
        self.height = 3  # Image height [pixels]
        self.width = 3  # Image width [pixels]
        self.timeStamp = 0  # Time stamp [s] */
        self.format = ""  # Image format string (RGB8, BGR,...)
        self.data = np.zeros((self.height, self.width, 3), np.uint8)  # The image data itself
        self.data.shape = self.height, self.width, 3

    def __str__(self):
        return f"Image:" \
               f"\nHeight: {self.height}\nWidth: {self.width}\n" \
               f"Format: {self.format}\nTimeStamp: {self.timeStamp}\nData: {self.data}"

    @staticmethod
    def image_msg_to_image(img, cv_image):

        image.width = img.width
        image.height = img.height
        image.format = "RGB8"
        image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        image.data = cv_image

        return image