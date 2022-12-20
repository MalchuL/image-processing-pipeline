import cv2
import numpy as np

from pipeline.postprocess.merging_crops.merging_crops import MergingCrops


class SeamlessMergingCrops(MergingCrops):
    def _merge_crop(self, crop_image, bbox, full_image):
        x1, y1, x2, y2 = bbox
        W, H = x2 - x1, y2 - y1
        center = round((x2 + x1) / 2), round((y2 + y1) / 2)
        crop_image = cv2.resize(crop_image, (W, H), interpolation=cv2.INTER_LINEAR)
        crop_mask = cv2.resize(self.mask, (W, H), interpolation=cv2.INTER_LINEAR)
        full_image = cv2.seamlessClone(crop_image, full_image, (crop_mask > 0.0).astype(np.uint8) * 255,
                                       center, cv2.NORMAL_CLONE)
        return full_image
