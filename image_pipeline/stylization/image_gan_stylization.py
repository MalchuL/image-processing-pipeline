from abc import ABC, abstractmethod

import cv2
import numpy as np

from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.stylization.gan_stylization import GANStylization
from image_pipeline.stylization.inference_engine.infer import InferenceEngine


class ImageGANStylization(GANStylization):
    """Pipeline task for crops stylization"""

    def _get_target(self, data: ImagePipelineData) -> np.ndarray:
        return data.processed_image

    def _postprocess(self, out_data: np.ndarray, data: ImagePipelineData) -> ImagePipelineData:
        data.processed_image = out_data
        return data
