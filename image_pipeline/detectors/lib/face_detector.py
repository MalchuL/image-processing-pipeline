from abc import ABC, abstractmethod
from typing import List

import numpy as np

class FaceDetector(ABC):
    def __init__(self, target_size):
        self.detector = self._get_detector()
        self.target_size = target_size

    @abstractmethod
    def _get_detector(self):
        pass

    @abstractmethod
    def _get_detection_outputs(self, img):
        pass

    @abstractmethod
    def _detect_crops(self, img, *args, **kwargs) -> List[np.ndarray]:
        """
        Img is a numpy ndarray in range [0..255], uint8 dtype, RGB type
        Returns ndarray with [x1, y1, x2, y2] in row
        """
        pass

    @abstractmethod
    def _postprocess_crops(self, crops, *args, **kwargs) -> List[np.ndarray]:
        return crops

    def _sort_faces(self, crops):
        sorted_faces = sorted(crops, key=lambda x: -(x[2] - x[0]) * (x[3] - x[1]))
        sorted_faces = np.stack(sorted_faces, axis=0)
        return sorted_faces

    def _fix_range_crops(self, img, crops):
        H, W, _ = img.shape
        final_crops = []
        for crop in crops:
            x1, y1, x2, y2 = crop
            x1 = max(min(round(x1), W), 0)
            y1 = max(min(round(y1), H), 0)
            x2 = max(min(round(x2), W), 0)
            y2 = max(min(round(y2), H), 0)
            new_crop = [x1, y1, x2, y2]
            final_crops.append(new_crop)
        final_crops = np.array(final_crops, dtype=np.int32)
        return final_crops

    def _crop_faces(self, img, crops) -> (List[np.ndarray], np.ndarray):
        cropped_faces = []
        for crop in crops:
            x1, y1, x2, y2 = crop
            face_crop = img[y1:y2, x1:x2, :]
            cropped_faces.append(face_crop)
        return cropped_faces, crops

    def _unify_and_merge(self, cropped_images, crops):
        return cropped_images, crops

    def __call__(self, img):
        return self.detect_faces(img)

    def detect_faces(self, img):
        crops = self._detect_crops(img)
        if crops is None or len(crops) == 0:
            return [], []
        crops = self._sort_faces(crops)
        updated_crops = self._postprocess_crops(crops)
        updated_crops = self._fix_range_crops(img, updated_crops)
        cropped_faces, updated_crops = self._crop_faces(img, updated_crops)
        unified_faces, updated_crops = self._unify_and_merge(cropped_faces, updated_crops)
        return unified_faces, updated_crops

