import cv2
import threading
import rospy

import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageROS

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
        #self.config = QLearnConfig()

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


def imageMsg2Image(img, bridge):
    image = Image()
    image.width = img.width
    image.height = img.height
    image.format = "RGB8"
    image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
    cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    image.data = cv_image
    return image


class Image:
    def __init__(self):
        self.height = 3  # Image height [pixels]
        self.width = 3  # Image width [pixels]
        self.timeStamp = 0  # Time stamp [s] */
        self.format = ""  # Image format string (RGB8, BGR,...)
        self.data = np.zeros(
            (self.height, self.width, 3), np.uint8
        )  # The image data itself
        self.data.shape = self.height, self.width, 3

    def __str__(self):
        s = (
            "Image: {\n   height: "
            + str(self.height)
            + "\n   width: "
            + str(self.width)
        )
        s = s + "\n   format: " + self.format + "\n   timeStamp: " + str(self.timeStamp)
        s = s + "\n   data: " + str(self.data) + "\n}"
        return s


class ListenerCamera:
    def __init__(self, topic):
        self.topic = topic
        self.data = Image()
        self.sub = None
        self.lock = threading.Lock()
        self.total_frames = 0

        self.bridge = CvBridge()
        self.start()

    def __callback(self, img):
        self.total_frames += 1
        image = imageMsg2Image(img, self.bridge)

        self.lock.acquire()
        self.data = image
        self.lock.release()

    def stop(self):
        self.sub.unregister()

    def start(self):
        self.sub = rospy.Subscriber(self.topic, ImageROS, self.__callback)

    def getImage(self):
        self.lock.acquire()
        image = self.data
        self.lock.release()

        return image

    def getTopic(self):
        return self.topic

    def hasproxy(self):
        return hasattr(self, "sub") and self.sub
