import cv2

from image_pipeline.pipeline import Pipeline
from image_pipeline.processed_data import ImagePipelineData


class ReadOpenCVImage(Pipeline):
    """Pipeline task to capture images from directory"""

    def process(self, data: ImagePipelineData) -> ImagePipelineData:

        image_file = data.input_image_path
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.input_image = image
        data.processed_image = image.copy()
        return data

    def filter(self, data: ImagePipelineData) -> bool:
        if data.input_image is not None:
            return False
        return True
