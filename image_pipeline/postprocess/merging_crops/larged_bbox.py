from abc import abstractmethod

import cv2
import numpy as np

from image_pipeline.pipeline import Pipeline
from image_pipeline.processed_data import ImagePipelineData


class LargerBBox(Pipeline):

    def process(self, data: ImagePipelineData) -> ImagePipelineData:
        sorted_boxes = sorted(data.processed_detection_bboxes,
                              key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))  # Sort by area
        result_image = sorted_boxes[-1].crop  # Taking image with larger area
        data.processed_image = result_image
        return data

    def filter(self, data: ImagePipelineData) -> bool:
        if data.processed_detection_bboxes is None or len(data.processed_detection_bboxes) == 0:
            return False
        return True
