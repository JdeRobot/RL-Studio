import sys

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class ImageConverter:
    def __init__(self):

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (32, 32))
        # cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image window", 480, 480)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)


def main(args):
    ic = image_converter()
    rospy.init_node("image_converter", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
