import os.path
from pathlib import Path

import cv2
import numpy as np
import pytest

from image_pipeline.compose import Compose
from image_pipeline.detectors.face_cropping import FaceCropping
from image_pipeline.detectors.lib.mediapipe_detector import StatMediaPipeDetector
from image_pipeline.postprocess.merging_crops.seamless_merging_crops import SeamlessMergingCrops
from image_pipeline.preprocess.resize2division import Resize2Dividable
from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.readers.opencv_read_image import ReadOpenCVImage
from image_pipeline.stylization.bbox_gan_stylization import BBoxGANStylization
from image_pipeline.stylization.image_gan_stylization import ImageGANStylization
from image_pipeline.stylization.inference_engine.inference_engine_cache import get_onnx_inference
from image_pipeline.stylization.inference_engine.onnx_image_infer import ONNXImageInference
from image_pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG

MUST_DIVIDED = 32
TARGET_SHAPE = 256

@pytest.mark.parametrize('img_name', CONFIG.IMAGES + CONFIG.IMAGES_NO_FACES)
def test_merging_pipeline(datadir, img_name, tmpdir):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    read_node = ReadOpenCVImage()
    resizer_node = Resize2Dividable(must_divided=MUST_DIVIDED)
    detector = StatMediaPipeDetector(TARGET_SHAPE)
    cropping_node = FaceCropping(detector)
    model_path = Path(datadir) / CONFIG.ONNX_MODEL_PATH
    inference_engine1 = get_onnx_inference(model_path)
    image_stylization_node = ImageGANStylization(inference_engine1)
    inference_engine2 = get_onnx_inference(model_path)
    crops_stylization_node = BBoxGANStylization(inference_engine2)
    assert inference_engine1 is inference_engine2
    merging_node = SeamlessMergingCrops(TARGET_SHAPE, debug=True)
    compose_node = Compose([read_node, resizer_node, cropping_node, image_stylization_node, crops_stylization_node, merging_node])

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    if img_name in CONFIG.IMAGES_NO_FACES:
        assert not merging_node.filter(data)
    else:
        assert merging_node.filter(data)
    tmpdir = str(tmpdir)
    result_image = result.processed_image
    print(result_image.shape)
    h, w, c = result_image.shape
    assert h % MUST_DIVIDED == 0
    assert w % MUST_DIVIDED == 0
    assert c == 3

    cv2.imwrite(os.path.join(tmpdir, img_name), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))


@pytest.mark.parametrize('img_name', CONFIG.IMAGES + CONFIG.IMAGES_NO_FACES)
def test_no_detection_merging_pipeline(datadir, img_name, tmpdir):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    read_node = ReadOpenCVImage()
    resizer_node = Resize2Dividable(must_divided=MUST_DIVIDED)
    model_path = Path(datadir) / CONFIG.ONNX_MODEL_PATH
    inference_engine = ONNXImageInference(model_path)
    image_stylization_node = ImageGANStylization(inference_engine)
    crops_stylization_node = BBoxGANStylization(inference_engine)
    merging_node = SeamlessMergingCrops(TARGET_SHAPE, debug=True)
    compose_node = Compose([read_node, resizer_node, image_stylization_node, crops_stylization_node, merging_node])

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    assert not merging_node.filter(data)
    tmpdir = str(tmpdir)
    result_image = result.processed_image
    print(result_image.shape)
    h, w, c = result_image.shape
    assert h % MUST_DIVIDED == 0
    assert w % MUST_DIVIDED == 0
    assert c == 3

    cv2.imwrite(os.path.join(tmpdir, img_name), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))


