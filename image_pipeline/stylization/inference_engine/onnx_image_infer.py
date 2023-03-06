from typing import Union

import numpy as np

from image_pipeline.stylization.inference_engine.infer import InferenceEngine
from image_pipeline.stylization.inference_engine.models.onnx_model import ONNXModel


class ONNXImageInference(InferenceEngine):
    MAX_IMG_VALUE = 255

    def __init__(self, model_path, mean: Union[float, np.ndarray] = 0.5, std: Union[float, np.ndarray] = 0.5, onnx_providers=None, chunk_size=4):
        if isinstance(mean, np.ndarray) and len(mean) == 3:
            mean = np.expand_dims(mean, 0)
        if isinstance(std, np.ndarray) and len(std) == 3:
            std = np.expand_dims(std, 0)
        self.mean = mean
        self.std = std
        self.model = ONNXModel(model_path, providers=onnx_providers)
        self.chunk_size = chunk_size

    def _normalize(self, img):
        return (img / self.MAX_IMG_VALUE - self.mean) / self.std

    def _denormalize(self, img):
        return (img * self.std + self.mean) * self.MAX_IMG_VALUE

    def forward(self, img):
        is_batch = True
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
            is_batch = False
        num_iters = (img.shape[0] + self.chunk_size - 1) // self.chunk_size
        out_images = []
        for i in range(num_iters):
            img_batch = img[i * self.chunk_size: (i + 1) * self.chunk_size]
            img_batch = self._normalize(img_batch)
            img_batch = img_batch.transpose(0, 3, 1, 2)
            out_image = self.model(img_batch)
            out_image = out_image.transpose(0, 2, 3, 1)
            out_image = self._denormalize(out_image)
            out_image = np.clip(out_image, 0, self.MAX_IMG_VALUE).astype(np.uint8)
            out_images.append(out_image)
        out_image = np.concatenate(out_images, axis=0)
        assert out_image.shape[0] == img.shape[0]
        if not is_batch:
            out_image = out_image[0]
        return out_image
