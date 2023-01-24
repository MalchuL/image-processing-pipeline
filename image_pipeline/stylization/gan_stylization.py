from abc import ABC, abstractmethod

import numpy as np

from image_pipeline.pipeline import Pipeline
from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.stylization.inference_engine.infer import InferenceEngine


class GANStylization(Pipeline, ABC):
    """Pipeline task for crops stylization"""

    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine

    @abstractmethod
    def _get_target(self, data: ImagePipelineData) -> np.ndarray:
        pass

    @abstractmethod
    def _postprocess(self, out_data: np.ndarray, data: ImagePipelineData) -> ImagePipelineData:
        pass

    def process(self, data: ImagePipelineData):
        input_data = self._get_target(data)
        out_data = self.inference_engine(input_data)
        out_data = self._postprocess(out_data, data)
        return out_data
