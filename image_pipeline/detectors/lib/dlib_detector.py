import logging

import cv2
import numpy as np

from .box_utils import convert_to_square
from .face_detector import FaceDetector

try:
    import dlib
except:
    logging.debug('Cannot import dlib')


class StatDlibFaceDetector(FaceDetector):
    def __init__(self, target_size, must_divide=32):
        super().__init__(target_size)
        # self.relative_offsets = [0.3258, 0.5225, 0.3258, 0.1290]  # Calculated from  calculate_aligned_offsets.ipynb wothout deviation
        self.must_divide = must_divide
        self.relative_offsets = [0.3619, 0.5830, 0.3619, 0.1909]  # Calculated from  calculate_aligned_offsets.ipynb

    def _get_detector(self):
        return dlib.get_frontal_face_detector()

    def _get_detection_outputs(self, img):
        return self.detector(img, 1)

    def _postprocess_crops(self, crops, *args, **kwargs) -> np.ndarray:
        final_crops = []
        x1_offset, y1_offset, x2_offset, y2_offset = self.relative_offsets
        for crop in crops:
            x1, y1, x2, y2 = [value // self.must_divide * self.must_divide for value in crop]
            w, h = x2 - x1, y2 - y1
            x1 -= w * x1_offset
            y1 -= h * y1_offset
            x2 += w * x2_offset
            y2 += h * y2_offset
            crop = np.array([x1, y1, x2, y2], dtype=crop.dtype)
            crop = convert_to_square(crop)
            final_crops.append(crop)
        final_crops = np.stack(final_crops, axis=0)
        return final_crops

    def _detect_crops(self, img, *args, **kwargs):
        faces = self._get_detection_outputs(img)
        crops = []
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            crop = np.array([x1, y1, x2, y2])
            crops.append(crop)
        if len(crops) > 0:
            crops = np.stack(crops, axis=0)
        return crops

    def _unify_and_merge(self, cropped_images, crops):
        if self.target_size is None:
            return cropped_images, crops
        else:
            resized_images = []
            for cropped_image in cropped_images:
                resized_image = cv2.resize(cropped_image, (self.target_size, self.target_size),
                                           interpolation=cv2.INTER_LINEAR)
                resized_images.append(resized_image)

            resized_images = np.stack(resized_images, axis=0)
            return resized_images, crops
