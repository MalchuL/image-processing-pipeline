from pathlib import Path


import pytest

from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.readers.opencv_read_image import ReadOpenCVImage
from image_pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
def test_capture_images(datadir, img_name):
    node = ReadOpenCVImage()
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = node(data)
    assert result.processed_image is not None
    assert result.input_image is not None
    assert result.processed_image.shape == result.input_image.shape
    print('Result shape', result.processed_image.shape)
