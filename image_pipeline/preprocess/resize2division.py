import numpy as np

from image_pipeline.utils.errors import PipelineConfigurationError
from image_pipeline.pipeline import Pipeline
from image_pipeline.processed_data import ImagePipelineData


class Resize2Dividable(Pipeline):
    """Pipeline task to crop image which shape must divide by input value (Usefull for Neural Networks which cannot works with random shape)"""

    def __init__(self, must_divided=32):
        assert must_divided % 2 == 0
        self._must_divided = must_divided

    @staticmethod
    def _get_center_crop_coords(height: int, width: int, crop_height: int, crop_width: int):
        y1 = (height - crop_height) // 2
        y2 = y1 + crop_height
        x1 = (width - crop_width) // 2
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    @staticmethod
    def _center_crop(img: np.ndarray, crop_height: int, crop_width: int):
        height, width = img.shape[:2]
        if height < crop_height or width < crop_width:
            raise ValueError(
                "Requested crop size ({crop_height}, {crop_width}) is "
                "larger than the image size ({height}, {width})".format(
                    crop_height=crop_height, crop_width=crop_width, height=height, width=width
                )
            )
        x1, y1, x2, y2 = Resize2Dividable._get_center_crop_coords(height, width, crop_height, crop_width)
        img = img[y1:y2, x1:x2]
        return img

    def process(self, data: ImagePipelineData):
        img = data.processed_image

        h, w, c = img.shape
        h = h // self._must_divided * self._must_divided
        w = w // self._must_divided * self._must_divided
        if h == 0 or w == 0:
            raise PipelineConfigurationError(
                f'Result shapes is 0 along dimension. New shapes is {h, w, c}, because your divided shape is {self._must_divided}. Image shape is {img.shape}')
        img = self._center_crop(img, h, w)

        data.processed_image = img
        return data
