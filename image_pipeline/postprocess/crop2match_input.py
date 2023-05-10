import numpy as np

from image_pipeline.pipeline import Pipeline
from image_pipeline.processed_data import ImagePipelineData


class Crop2MatchInput(Pipeline):
    """Pipeline task to crop image which shape must be same as input"""

    @staticmethod
    def _crop_image_by_offsets(img: np.ndarray, top: int, left: int, bottom: int, right: int):
        h, w, c = img.shape
        return img[top:h - bottom, left:w - right, :]

    def process(self, data: ImagePipelineData):
        result_image = data.processed_image
        t, l, b, r = data.additional_kwargs.output_cropping
        result_image = self._crop_image_by_offsets(img=result_image, top=t, left=l, bottom=b, right=r)
        data.processed_image = result_image
        return data

    def filter(self, data: ImagePipelineData) -> bool:
        return data.additional_kwargs.output_cropping is not None and len(data.additional_kwargs.output_cropping) == 4
