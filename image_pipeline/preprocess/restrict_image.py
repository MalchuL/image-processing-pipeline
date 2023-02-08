import cv2
import numpy as np

from image_pipeline.utils.errors import PipelineConfigurationError
from image_pipeline.pipeline import Pipeline
from image_pipeline.processed_data import ImagePipelineData


class RestrictImage(Pipeline):
    """Pipeline task to crop image which shape must divide by input value (Usefull for Neural Networks which cannot works with random shape)"""

    def __init__(self, max_size=1920):
        self.max_size = max_size

    @staticmethod
    def _image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    def process(self, data: ImagePipelineData):
        img = data.processed_image

        h, w, c = img.shape

        if h == 0 or w == 0:
            raise PipelineConfigurationError(
                f'Result shapes is 0 along dimension.')

        if h > w:
            if h > self.max_size:
                img = self._image_resize(img, height=self.max_size)
        else:
            if w > self.max_size:
                img = self._image_resize(img, width=self.max_size)

        data.processed_image = img
        return data
