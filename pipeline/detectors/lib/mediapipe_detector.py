import logging

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
