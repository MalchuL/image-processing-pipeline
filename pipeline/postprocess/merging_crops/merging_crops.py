from abc import abstractmethod

import cv2
import numpy as np

from pipeline.pipeline import Pipeline
from pipeline.processed_data import ImagePipelineData


class MergingCrops(Pipeline):
    def __init__(self, crops_size: int, debug=False):
        self.mask = self._create_circular_mask(crops_size, crops_size)

        self.debug = debug

    def process(self, data: ImagePipelineData) -> ImagePipelineData:
        result_image = data.processed_image
        for detecton in data.processed_detection_bboxes:
            bbox = detecton.bbox
            crop = detecton.crop
            result_image = self._merge_crop(crop, bbox, result_image)
            if self.debug:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0))
        data.processed_image = result_image
        return data

    @abstractmethod
    def _merge_crop(self, crop_image, bbox, full_image):
        pass

    @staticmethod
    def _create_circular_mask(h, w, power=None, clipping_coef=0.85):
        center = (int(w / 2), int(h / 2))

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        print(dist_from_center.max(), dist_from_center.min())
        clipping_radius = min((h - center[0]), (w - center[1])) * clipping_coef
        max_size = max((h - center[0]), (w - center[1]))
        dist_from_center[dist_from_center < clipping_radius] = clipping_radius
        dist_from_center[dist_from_center > max_size] = max_size
        max_distance, min_distance = np.max(dist_from_center), np.min(dist_from_center)
        dist_from_center = 1 - (dist_from_center - min_distance) / (max_distance - min_distance)
        if power is not None:
            dist_from_center = np.power(dist_from_center, power)
        dist_from_center = np.stack([dist_from_center] * 3, axis=2)
        # mask = dist_from_center <= radius
        return dist_from_center

    def filter(self, data: ImagePipelineData) -> bool:
        if data.processed_detection_bboxes is None or len(data.processed_detection_bboxes) == 0:
            return False
        return True
