from typing import Union

import numpy as np

from pipeline.stylization.inference_engine.infer import InferenceEngine
from pipeline.stylization.inference_engine.models.onnx_model import ONNXModel


class ONNXImageInference(InferenceEngine):
    MAX_IMG_VALUE = 255

    def __init__(self, model_path, mean: Union[float, np.ndarray] = 0.5, std: Union[float, np.ndarray] = 0.5):
        if isinstance(mean, np.ndarray) and len(mean) == 3:
            mean = np.expand_dims(mean, 0)
        if isinstance(std, np.ndarray) and len(std) == 3:
            std = np.expand_dims(std, 0)
        self.mean = mean
        self.std = std
        self.model = ONNXModel(model_path)

    def _normalize(self, img):
        return (img / self.MAX_IMG_VALUE - self.mean) / self.std

    def _denormalize(self, img):
        return (img * self.std + self.mean) * self.MAX_IMG_VALUE

    def forward(self, img):
        is_batch = True
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
            is_batch = True
        img = self._normalize(img)
        img = img.transpose(0, 3, 1, 2)
        out_image = self.model(img)
        out_image = out_image.transpose(0, 2, 3, 1)
        out_image = self._denormalize(out_image)
        out_image = np.clip(out_image, 0, self.MAX_IMG_VALUE).astype(np.uint8)
        if not is_batch:
            out_image = out_image[0]
        return out_image
