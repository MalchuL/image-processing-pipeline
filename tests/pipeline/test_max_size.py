from pathlib import Path

import pytest

from image_pipeline.preprocess.restrict_image import RestrictImage
from image_pipeline.utils.errors import PipelineConfigurationError
from image_pipeline.compose import Compose
from image_pipeline.preprocess.resize2division import Resize2Dividable
from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.readers.opencv_read_image import ReadOpenCVImage
from image_pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG

RAISES_DIVIDED = 10_000


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
@pytest.mark.parametrize("max_size", [256, 112])
def test_max_size_node(datadir, img_name, max_size):
    read_node = ReadOpenCVImage()
    resize_node = RestrictImage(max_size=max_size)
    compose_node = Compose([read_node, resize_node])

    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    assert result.processed_image is not None
    assert result.input_image is not None
    assert result.processed_image.shape != result.input_image.shape
    h, w, _ = result.processed_image.shape
    assert h <= max_size
    assert w <= max_size
    print('Result shape', result.processed_image.shape, 'Input', result.processed_image.shape)

