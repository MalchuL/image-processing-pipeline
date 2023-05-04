import os
from pathlib import Path

import cv2
import pytest

from image_pipeline.postprocess.crop2match_input import Crop2MatchInput
from image_pipeline.preprocess.pad2division import Pad2Dividable
from image_pipeline.utils.errors import PipelineConfigurationError
from image_pipeline.compose import Compose
from image_pipeline.preprocess.resize2division import Resize2Dividable
from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.readers.opencv_read_image import ReadOpenCVImage
from image_pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG

RAISES_DIVIDED = 10_000


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
@pytest.mark.parametrize("must_divided", [32, 64])
def test_padding_node(tmpdir, datadir, img_name, must_divided):
    read_node = ReadOpenCVImage()
    padding_node = Pad2Dividable(must_divided=must_divided)
    compose_node = Compose([read_node, padding_node])

    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    assert result.processed_image is not None
    assert result.input_image is not None
    assert result.processed_image.shape != result.input_image.shape
    h, w, _ = result.processed_image.shape
    assert h % must_divided == 0
    assert w % must_divided == 0
    input_h, input_w, _ = result.input_image.shape
    assert h >= input_h
    assert w >= input_w
    assert data.additional_kwargs.output_cropping is not None
    assert len(data.additional_kwargs.output_cropping) == 4
    print('Result shape', result.processed_image.shape, 'Input', result.input_image.shape)
    cv2.imwrite(os.path.join(tmpdir, f'{img_name}'), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))


@pytest.mark.parametrize('img_name', [CONFIG.IMAGES[1]])
@pytest.mark.parametrize("must_divided", [64])
@pytest.mark.parametrize("use_pad", [True, False])
def test_padding_and_cropping_node(tmpdir, datadir, img_name, must_divided, use_pad):
    read_node = ReadOpenCVImage()
    padding_node = Pad2Dividable(must_divided=must_divided)
    cropping_node = Crop2MatchInput()
    if use_pad:
        compose_node = Compose([read_node, padding_node, cropping_node])
    else:
        compose_node = Compose([read_node, cropping_node])

    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    assert result.processed_image is not None
    assert result.input_image is not None
    assert result.processed_image.shape == result.input_image.shape


@pytest.mark.parametrize('img_name', [CONFIG.IMAGES[0]])
@pytest.mark.parametrize("must_divided", [64])
@pytest.mark.parametrize("use_pad", [True, False])
def test_padding_and_cropping_node(tmpdir, datadir, img_name, must_divided, use_pad):
    read_node = ReadOpenCVImage()
    padding_node = Pad2Dividable(must_divided=must_divided)
    cropping_node = Crop2MatchInput()
    compose_node = Compose([read_node, padding_node, cropping_node])


    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    data.additional_kwargs.img_padding['skip_padding'] = not use_pad  # Possible keys: `skip_padding`
    result: ImagePipelineData = compose_node(data)
    assert result.processed_image is not None
    assert result.input_image is not None
    if use_pad:
        assert result.processed_image.shape == result.input_image.shape
    else:
        assert result.processed_image.shape != result.input_image.shape

