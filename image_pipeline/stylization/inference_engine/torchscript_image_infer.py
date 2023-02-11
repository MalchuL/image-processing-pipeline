from image_pipeline.stylization.inference_engine.models.torchscript_model import TorchScriptModel
from image_pipeline.stylization.inference_engine.onnx_image_infer import ONNXImageInference
from typing import Union

import numpy as np


class TorchScriptImageInference(ONNXImageInference):
    def __init__(self, model_path, mean: Union[float, np.ndarray] = 0.5, std: Union[float, np.ndarray] = 0.5,
                 use_gpu=False):
        if isinstance(mean, np.ndarray) and len(mean) == 3:
            mean = np.expand_dims(mean, 0)
        if isinstance(std, np.ndarray) and len(std) == 3:
            std = np.expand_dims(std, 0)
        self.mean = mean
        self.std = std
        self.model = TorchScriptModel(model_path, use_cuda=use_gpu)
