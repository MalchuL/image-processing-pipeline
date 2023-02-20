import logging
from typing import List

import cv2
import numpy as np

from .dlib_detector import StatDlibFaceDetector

try:
    import mediapipe as mp
except:
    logging.debug('Cannot import dlib')


class StatMediaPipeDetector(StatDlibFaceDetector):
    def __init__(self, target_size, must_divide=32):
        super().__init__(target_size, must_divide)
        self.relative_offsets = [0.3209, 0.4509, 0.3209, 0.2052]

    def _get_detector(self):
        mp_face_detection = mp.solutions.face_detection
        detector = None
        return detector

    def _get_detection_outputs(self, img):
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
            outputs = face_detector.process(img)
        return outputs.detections

    def _detect_crops(self, img, *args, **kwargs):
        faces = self._get_detection_outputs(img)
        h, w, _ = img.shape
        crops = []
        if faces is None:
            faces = np.zeros((0, 4))
        for face in faces:
            bbox = face.location_data.relative_bounding_box
            x1 = bbox.xmin * w
            y1 = bbox.ymin * h
            x2 = (bbox.xmin + bbox.width) * w
            y2 = (bbox.ymin + bbox.height) * h
            crop = np.array([x1, y1, x2, y2])
            crops.append(crop)
        if len(crops) > 0:
            crops = np.stack(crops, axis=0)
        return crops


class ScaledStatMediaPipeDetector(StatMediaPipeDetector):
    def __init__(self, target_size, must_divide=32, scale_bbox=1.0):
        super().__init__(target_size, must_divide)
        self.scale_bbox = scale_bbox

    def _crop_faces(self, img, crops) -> (List[np.ndarray], np.ndarray):
        cropped_faces = []
        H, W, _ = img.shape
        new_crops = []

        for crop in crops:
            x1, y1, x2, y2 = crop
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            wh = ((x2 - x1), (y2 - y1))
            new_wh = (wh[0] * self.scale_bbox, wh[1] * self.scale_bbox)

            x1 = max(min(round(center[0] - new_wh[0] / 2), W), 0)
            y1 = max(min(round(center[1] - new_wh[1] / 2), H), 0)
            x2 = max(min(round(center[0] + new_wh[0] / 2), W), 0)
            y2 = max(min(round(center[1] + new_wh[1] / 2), H), 0)

            new_wh = ((x2 - x1), (y2 - y1))
            face_crop = img[y1:y2, x1:x2, :]

            face_crop = cv2.resize(face_crop,
                                   ((round(self.target_size * new_wh[0] / wh[0]) + self.must_divide - 1) // self.must_divide * self.must_divide,
                                    (round(self.target_size * new_wh[1] / wh[1]) + self.must_divide - 1)// self.must_divide * self.must_divide),
                                   interpolation=cv2.INTER_LINEAR)
            cropped_faces.append(face_crop)
            new_crops.append([x1, y1, x2, y2])
        return cropped_faces, new_crops

    def _unify_and_merge(self, cropped_images, crops):
        return cropped_images, crops


class AllScaledStatMediaPipeDetector(ScaledStatMediaPipeDetector):

    def _unify_and_merge(self, cropped_images, crops):
        if self.target_size is None:
            return cropped_images, crops
        else:
            resized_images = []
            for cropped_image in cropped_images:
                resized_image = cv2.resize(cropped_image,
                                           ((round(
                                               self.target_size * self.scale_bbox) + self.must_divide - 1) // self.must_divide * self.must_divide,
                                            (round(
                                                self.target_size * self.scale_bbox) + self.must_divide - 1) // self.must_divide * self.must_divide),
                                           interpolation=cv2.INTER_LINEAR)
                resized_images.append(resized_image)

            resized_images = np.stack(resized_images, axis=0)
            return resized_images, crops
