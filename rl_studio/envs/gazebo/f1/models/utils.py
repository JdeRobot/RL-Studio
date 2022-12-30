import cv2


class F1GazeboUtils:
    def __init__(self):
        self.f1 = None

    @staticmethod
    def show_image_with_centrals(
        name, img, waitkey, centrals_normalized, centrals_in_pixels, x_row
    ):
        window_name = f"{name}"

        for index, value in enumerate(x_row):
            cv2.putText(
                img,
                str(
                    f"{int(centrals_in_pixels[index])} - [{int(centrals_normalized[index])}]"
                ),
                (int(centrals_in_pixels[index]), int(x_row[index])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)

    def show_image(self, name, img, waitkey):
        window_name = f"{name}"
        cv2.imshow(window_name, img)
        cv2.waitKey(waitkey)
