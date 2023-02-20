import cv2
import numpy as np

from image_pipeline.detectors.lib.face_detector import FaceDetector
from image_pipeline.pipeline import Pipeline
from image_pipeline.processed_data import ImagePipelineData, DetectionResult


class FaceCropping(Pipeline):
    """Pipeline task to crop image which shape must divide by input value (Usefull for Neural Networks which cannot works with random shape)"""

    def __init__(self, detector: FaceDetector):
        self.detector = detector

    def process(self, data: ImagePipelineData):
        img = data.processed_image
        if data.additional_kwargs is None:
            kwargs = {}
        else:
            if data.additional_kwargs.detection is not None:
                kwargs = data.additional_kwargs.detection
            else:
                kwargs = {}
        images, crops = self.detector(img, **kwargs)
        crops = [DetectionResult(crop, image) for image, crop in zip(images, crops)]
        data.detection_bboxes = crops
        data.processed_detection_bboxes = crops.copy()
        return data
