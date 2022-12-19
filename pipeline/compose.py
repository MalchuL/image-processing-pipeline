from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np

from pipeline.pipeline import Pipeline
from pipeline.processed_data import ImagePipelineData


# img: Union[str, np.ndarray, List[str, np.ndarray]]
class Compose(Pipeline):
    """Common pipeline class fo all pipeline tasks."""

    def __init__(self, nodes: List[Pipeline]):
        self.nodes = nodes

    def process(self, data: ImagePipelineData):
        out_data = data
        for node in self.nodes:
            out_data = node(out_data)
        return out_data
