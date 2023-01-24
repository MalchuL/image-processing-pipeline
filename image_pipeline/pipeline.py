from abc import ABC, abstractmethod

from image_pipeline.processed_data import ImagePipelineData


class Pipeline(ABC):
    """Common pipeline class fo all pipeline tasks."""

    @abstractmethod
    def process(self, data: ImagePipelineData) -> ImagePipelineData:
        pass

    def filter(self, data: ImagePipelineData) -> bool:
        """It might be better to fast skip process"""

        return True

    def __call__(self, data: ImagePipelineData) -> ImagePipelineData:
        return self.process(data)
