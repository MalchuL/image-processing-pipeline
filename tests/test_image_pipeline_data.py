import logging

import numpy as np

from pipeline.processed_data import ImagePipelineData, DetectionResult


def test_data():
    data1 = ImagePipelineData(input_image_path='sdf')
    data2 = ImagePipelineData()
    logging.debug(data1)
    logging.debug(data2)
