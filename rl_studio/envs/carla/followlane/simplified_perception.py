import cv2
import numpy as np

from rl_studio.envs.carla.followlane.utils import AutoCarlaUtils


class AutoCarlaSimplifiedPerception:
    def preprocess_image(self, img):
        """
        image is trimming from top to middle
        """
        ## first, we cut the upper image
        height = img.shape[0]
        image_middle_line = (height) // 2
        img_sliced = img[image_middle_line:]
        ## calculating new image measurements
        # height = img_sliced.shape[0]
        ## -- convert to GRAY
        gray_mask = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        ## --  apply mask to convert in Black and White
        theshold = 50
        # _, white_mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)
        _, white_mask = cv2.threshold(gray_mask, theshold, 255, cv2.THRESH_BINARY)

        return white_mask

    def calculate_right_line(self, mask, x_row):
        """
        calculates distance from center to right line
        This distance will be using as a error from center lane
        """
        ## get total lines in every line point
        lines = [mask[x_row[i], :] for i, _ in enumerate(x_row)]

        ### ----------------- from right to left
        lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        # print(f"{lines_inversed = }")
        inv_index_right = [
            np.argmax(lines_inversed[x]) for x, _ in enumerate(lines_inversed)
        ]
        # print(f"{inv_index_right = }")
        # offset = 10
        # inv_index_right_plus_offset = [
        #    inv_index_right[x] + offset if inv_index_right[x] != 0 else 0
        #    for x, _ in enumerate(inv_index_right)
        # ]
        # print(f"{inv_index_right = }")
        # index_right = [
        #    mask.shape[1] - inv_index_right_plus_offset[x]
        #    if inv_index_right_plus_offset[x] != 0
        #    else 0
        #    for x, _ in enumerate(inv_index_right_plus_offset)
        # ]
        index_right = [
            mask.shape[1] - inv_index_right[x] if inv_index_right[x] != 0 else 0
            for x, _ in enumerate(inv_index_right)
        ]

        return index_right

    def resize_and_trimming_right_line_mask(self, img):
        """
        Only working for taking points wuth Right Line

        1. Dilate the points in the mask
        2. Trimming the left part in the image, to the center

        """
        kernel = np.ones((3, 3), np.uint8)
        img_dilation = cv2.dilate(img, kernel, iterations=2)
        # AutoCarlaUtils.show_image(
        #    "img_dilation",
        #    img_dilation,
        #    600,
        #    400,
        # )
        # trimming and resizing mask
        height = img_dilation.shape[0]
        center_image = img_dilation.shape[1] // 2
        image_middle_line = (height - 20) // 2

        img_cropped_right = img_dilation[:, center_image:]
        # AutoCarlaUtils.show_image(
        #    "img_sliced",
        #    img_sliced,
        #    600,
        #    700,
        # )
        
        image_resize = cv2.resize(img_dilation, (32, 32), cv2.INTER_AREA)
        
        # trimming left part 
        #image_resize = cv2.resize(img_cropped_right, (32, 32), cv2.INTER_AREA)
        
        image_resize = np.expand_dims(image_resize, axis=2)

        # AutoCarlaUtils.show_image(
        #    "image_resize",
        #    image_resize,
        #    700,
        #    1000,
        # )
        return image_resize
