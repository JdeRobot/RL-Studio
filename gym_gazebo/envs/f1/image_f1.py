import cv2
import numpy as np

from cprint import cprint


class ImageF1:

    font = cv2.FONT_HERSHEY_COMPLEX

    def __init__(self):

        #cprint.warn(f"\n [ImageF1] -> ------- Enter ImageF1 ---------------\n")

        self.height = 3  # Image height [pixels]
        self.width = 3  # Image width [pixels]
        self.timeStamp = 0  # Time stamp [s] */
        self.format = ""  # Image format string (RGB8, BGR,...)
        self.data = np.zeros((self.height, self.width, 3), np.uint8)  # The image data itself
        self.data.shape = self.height, self.width, 3

        #cprint.ok(f"\n  [ImageF1] -> -------- Out ImageF1 (__init__) -------------\n")


    def __str__(self):
        return f"Image:" \
               f"\nHeight: {self.height}\nWidth: {self.width}\n" \
               f"Format: {self.format}\nTimeStamp: {self.timeStamp}\nData: {self.data}"

#    @staticmethod
#    def image_msg_to_image(img, cv_image):
#        print(f"\n ImageF1.image_msg_to_image()\n")
#        image.width = img.width
#        image.height = img.height
#        image.format = "RGB8"
#        image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
#        image.data = cv_image

#        return image


    @staticmethod
    def show_telemetry(img, points, action, reward):
        print(f"\n ImageF1.show_telemetry()\n")
        count = 0
        for idx, point in enumerate(points):
            cv2.line(img, (320, x_row[idx]), (320, x_row[idx]), (255, 255, 0), thickness=5)
            # cv2.line(img, (center_image, x_row[idx]), (point, x_row[idx]), (255, 255, 255), thickness=2)
            cv2.putText(img, str("err{}: {}".format(idx+1, center_image - point)), (18, 340 + count), font, 0.4,
                        (255, 255, 255), 1)
            count += 20
        cv2.putText(img, str("action: {}".format(action)), (18, 280), font, 0.4, (255, 255, 255), 1)
        cv2.putText(img, str("reward: {}".format(reward)), (18, 320), font, 0.4, (255, 255, 255), 1)

        cv2.imshow("Image window", img[240:])
        cv2.waitKey(3)
