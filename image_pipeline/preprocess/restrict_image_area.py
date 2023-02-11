import math

import cv2
import numpy as np

from image_pipeline.utils.errors import PipelineConfigurationError
from image_pipeline.pipeline import Pipeline
from image_pipeline.processed_data import ImagePipelineData


class RestrictImageArea(Pipeline):
    """Pipeline task to crop image which shape must divide by input value (Usefull for Neural Networks which cannot works with random shape)"""

    def __init__(self, max_area=1920 * 1080):
        self.max_area = max_area

    @staticmethod
    def _image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        resized = cv2.resize(image, (width, height), interpolation=inter)

        # return the resized image
        return resized

    def process(self, data: ImagePipelineData):
        img = data.processed_image

        h, w, c = img.shape

        if h == 0 or w == 0:
            raise PipelineConfigurationError(
                f'Result shapes is 0 along dimension.')
        area = h * w
        if area > self.max_area:
            scale = math.sqrt(area / self.max_area)
            new_h = round(h / scale)
            new_w = round(w / scale)

            img = self._image_resize(img, height=new_h, width=new_w)

        data.processed_image = img
        return data
