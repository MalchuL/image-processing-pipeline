import logging

import numpy as np

from image_pipeline.preprocess.resize2division import Resize2Dividable
from image_pipeline.processed_data import ImagePipelineData
import cv2


class Pad2Dividable(Resize2Dividable):
    """Pipeline task to pad image which shape must divide by input value. If skip_padding is active will crop image"""
    ADDITIONAL_ARG = 'skip_padding'

    def __init__(self, must_divided=32, padding_mode=cv2.BORDER_REFLECT):
        super().__init__(must_divided)
        self._padding_mode = padding_mode

    @staticmethod
    def _bottom_right_pad(img: np.ndarray, padding_h, padding_w, padding_mode=cv2.BORDER_CONSTANT):
        img = cv2.copyMakeBorder(img, 0, padding_h, 0, padding_w, padding_mode)
        return img

    def process(self, data: ImagePipelineData):
        if data.additional_kwargs.img_padding is not None and data.additional_kwargs.img_padding.get(self.ADDITIONAL_ARG, False):
            return super().process(data)

        img = data.processed_image

        h, w, c = img.shape
        new_h = (h + self._must_divided - 1) // self._must_divided * self._must_divided
        new_w = (w + self._must_divided - 1) // self._must_divided * self._must_divided
        padding_h = new_h - h
        padding_w = new_w - w

        img = self._bottom_right_pad(img, padding_h=padding_h, padding_w=padding_w, padding_mode=self._padding_mode)

        data.processed_image = img
        if data.additional_kwargs.output_cropping is not None:
            logging.warning("output_cropping in image pipeline already defined, override it to new value")
        data.additional_kwargs.output_cropping = [0, 0, padding_h, padding_w]
        return data
