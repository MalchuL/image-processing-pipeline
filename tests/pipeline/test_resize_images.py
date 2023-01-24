from pathlib import Path

import numpy as np
import pytest

from errors import PipelineConfigurationError
from image_pipeline.compose import Compose
from image_pipeline.preprocess.resize2division import Resize2Dividable
from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.readers.opencv_read_image import ReadOpenCVImage
from image_pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG

RAISES_DIVIDED = 10_000


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
@pytest.mark.parametrize("must_divided", [32, 64])
def test_resize_node(datadir, img_name, must_divided):
    read_node = ReadOpenCVImage()
    resize_node = Resize2Dividable(must_divided=must_divided)
    compose_node = Compose([read_node, resize_node])

    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    assert result.processed_image is not None
    assert result.input_image is not None
    assert result.processed_image.shape != result.input_image.shape
    h, w, _ = result.processed_image.shape
    assert h % must_divided == 0
    assert w % must_divided == 0
    print('Result shape', result.processed_image.shape, 'Input', result.processed_image.shape)


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
def test_impossible_resize_node(datadir, img_name):
    read_node = ReadOpenCVImage()
    resize_node = Resize2Dividable(must_divided=RAISES_DIVIDED)
    compose_node = Compose([read_node, resize_node])

    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    with pytest.raises(PipelineConfigurationError):
        result: ImagePipelineData = compose_node(data)
