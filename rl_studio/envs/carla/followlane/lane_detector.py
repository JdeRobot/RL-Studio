import cv2
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as im
import rospy
from sensor_msgs.msg import Image as ImageROS
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import torch

from rl_studio.envs.gazebo.f1.image_f1 import ImageF1, ListenerCamera, Image
from rl_studio.envs.gazebo.f1.models.utils import F1GazeboUtils


class LaneDetector:
    def __init__(self, model_path):
        torch.cuda.empty_cache()
        self.model = torch.load(model_path)
        self.model.eval()

    def get_prediction(self, img_array):
        # def get_prediction(self, model, img_array):
        with torch.no_grad():
            image_tensor = img_array.transpose(2, 0, 1).astype("float32") / 255
            x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
            # back, left, right = (
            #   torch.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()[0]
            # )
            model_output = (
                torch.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()
            )

        # res, left_mask, right_mask = self.lane_detection_overlay(
        #   img_array, left, right
        # )

        # return res, left_mask, right_mask
        return model_output

    def lane_detection_overlay(self, image, left_mask, right_mask):
        res = np.copy(image)
        # We show only points with probability higher than 0.5
        res[left_mask > 0.07, :] = [255, 0, 0]
        res[right_mask > 0.07, :] = [0, 0, 255]

        # cv2.erode(left_mask, (7, 7), 4)
        # cv2.dilate(left_mask, (7, 7), 4)

        # return res, left_mask, right_mask
        return res
