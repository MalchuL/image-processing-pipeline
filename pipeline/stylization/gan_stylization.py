from abc import ABC, abstractmethod

import numpy as np

from pipeline.pipeline import Pipeline
from pipeline.processed_data import ImagePipelineData
from pipeline.stylization.inference_engine.infer import InferenceEngine


class GANStylization(Pipeline, ABC):
    """Pipeline task for crops stylization"""

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