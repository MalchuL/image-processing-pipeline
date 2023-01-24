from abc import ABC, abstractmethod

import cv2
import numpy as np

from image_pipeline.processed_data import ImagePipelineData, DetectionResult
from image_pipeline.stylization.gan_stylization import GANStylization
from image_pipeline.stylization.inference_engine.infer import InferenceEngine


class BBoxGANStylization(GANStylization):
    """Pipeline task for crops stylization"""

    def _get_target(self, data: ImagePipelineData) -> np.ndarray:
        images = np.stack([bbox.crop for bbox in data.processed_detection_bboxes], axis=0)
        return images

    def _postprocess(self, out_data: np.ndarray, data: ImagePipelineData) -> ImagePipelineData:
        out_bboxes = [DetectionResult(input_bbox.bbox, out_data_crop) for input_bbox, out_data_crop in
                      zip(data.processed_detection_bboxes, out_data)]
        data.processed_detection_bboxes = out_bboxes
        return data

    def filter(self, data: ImagePipelineData) -> bool:
        if data.processed_detection_bboxes is None or len(data.processed_detection_bboxes) == 0:
            return False
        return True
