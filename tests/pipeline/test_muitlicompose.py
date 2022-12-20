from pathlib import Path

import numpy as np
import pytest

from errors import PipelineConfigurationError
from pipeline.compose import Compose
from pipeline.preprocess.resize2division import Resize2Dividable
from pipeline.processed_data import ImagePipelineData
from pipeline.readers.opencv_read_image import ReadOpenCVImage
from pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG

RAISES_DIVIDED = 10_000


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
@pytest.mark.parametrize("must_divided1", [18, 14])
@pytest.mark.parametrize("must_divided2", [6, 8, 10])
def test_multi_resize_node(datadir, img_name, must_divided1, must_divided2):
    read_node = ReadOpenCVImage()
    resize_node_1 = Resize2Dividable(must_divided=must_divided1)
    resize_node_2 = Resize2Dividable(must_divided=must_divided2)
    compose_node = Compose([read_node, Compose([resize_node_1]), resize_node_2])

    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)



