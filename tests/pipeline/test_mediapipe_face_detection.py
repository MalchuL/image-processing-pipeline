import os.path
from pathlib import Path

import cv2
import pytest

from image_pipeline.compose import Compose
from image_pipeline.detectors.face_cropping import FaceCropping
from image_pipeline.detectors.lib.mediapipe_detector import StatMediaPipeDetector, AllScaledStatMediaPipeDetector
from image_pipeline.postprocess.merging_crops.seamless_merging_crops import SeamlessMergingCrops
from image_pipeline.preprocess.resize2division import Resize2Dividable
from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.readers.opencv_read_image import ReadOpenCVImage
from image_pipeline.stylization.bbox_gan_stylization import BBoxGANStylization
from image_pipeline.stylization.image_gan_stylization import ImageGANStylization
from image_pipeline.stylization.inference_engine.inference_engine_cache import get_onnx_inference
from image_pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG


MUST_DIVIDED = 32
TARGET_SHAPE = 256


@pytest.fixture()
def compose_node():
    read_node = ReadOpenCVImage()
    detector = StatMediaPipeDetector(256)
    cropping_node = FaceCropping(detector)
    compose_node = Compose([read_node, cropping_node])
    return compose_node

@pytest.mark.parametrize('img_name', CONFIG.IMAGES_NO_FACES)
def test_no_detection_crops(datadir, img_name, compose_node):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    assert result.processed_image is not None
    assert result.input_image is not None
    assert result.processed_image.shape == result.input_image.shape
    assert len(result.detection_bboxes) == len(result.processed_detection_bboxes) == 0

@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
def test_crops_detection(datadir, img_name, tmpdir, compose_node):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)

    assert result.processed_image is not None
    assert result.input_image is not None
    assert result.processed_image.shape == result.input_image.shape
    assert len(result.detection_bboxes) == len(result.processed_detection_bboxes)

    tmpdir = str(tmpdir)
    for i, (det_boxes, proc_boxes) in enumerate(zip(result.detection_bboxes, result.processed_detection_bboxes)):
        assert det_boxes == proc_boxes
        assert det_boxes.crop is not None
        assert det_boxes.crop.min() >= 0
        assert det_boxes.crop.max() <= 256

        h, w, c = result.processed_image.shape
        assert c == 3
        print(det_boxes.bbox)
        assert det_boxes.bbox[0] >= 0
        assert det_boxes.bbox[1] >= 0
        assert det_boxes.bbox[2] <= w
        assert det_boxes.bbox[3] <= h
        assert det_boxes.bbox[0] < det_boxes.bbox[2]
        assert det_boxes.bbox[1] < det_boxes.bbox[3]

        cv2.imwrite(os.path.join(tmpdir, str(i) + '_' + img_name), cv2.cvtColor(det_boxes.crop, cv2.COLOR_RGB2BGR))


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
@pytest.mark.parametrize('scale_bbox', [1.0, 2.0])
def test_larger_bbox_multi_pipeline(datadir, img_name, tmpdir, scale_bbox):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    read_node = ReadOpenCVImage()
    resizer_node = Resize2Dividable(must_divided=MUST_DIVIDED)
    detector = AllScaledStatMediaPipeDetector(TARGET_SHAPE, scale_bbox=1.2)
    cropping_node = FaceCropping(detector)
    model_path = Path(datadir) / CONFIG.ONNX_MODEL_PATH
    inference_engine = get_onnx_inference(model_path)
    image_stylization_node = ImageGANStylization(inference_engine)
    crops_stylization_node = BBoxGANStylization(inference_engine)

    merging_node = SeamlessMergingCrops(TARGET_SHAPE, debug=True)
    compose_node = Compose(
        [read_node, resizer_node, cropping_node, image_stylization_node, crops_stylization_node, merging_node])

    data = get_data(im_path)
    data.additional_kwargs.detection['scale_bbox'] = scale_bbox
    result: ImagePipelineData = compose_node(data)

    tmpdir = str(tmpdir)
    result_image = result.processed_image
    print(result_image.shape)
    h, w, c = result_image.shape
    assert h % MUST_DIVIDED == 0
    assert w % MUST_DIVIDED == 0
    assert c == 3

    cv2.imwrite(os.path.join(tmpdir, img_name), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
@pytest.mark.parametrize('scale_bbox', [1.0, 2.0])
def test_larger_bbox_multi_pipeline(datadir, img_name, tmpdir, scale_bbox):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    read_node = ReadOpenCVImage()
    resizer_node = Resize2Dividable(must_divided=MUST_DIVIDED)
    detector = StatMediaPipeDetector(TARGET_SHAPE)
    cropping_node = FaceCropping(detector)
    model_path = Path(datadir) / CONFIG.ONNX_MODEL_PATH
    inference_engine = get_onnx_inference(model_path)
    image_stylization_node = ImageGANStylization(inference_engine)
    crops_stylization_node = BBoxGANStylization(inference_engine)

    merging_node = SeamlessMergingCrops(TARGET_SHAPE, debug=True)
    compose_node = Compose(
        [read_node, resizer_node, cropping_node, image_stylization_node, crops_stylization_node, merging_node])

    data = get_data(im_path)
    data.additional_kwargs.detection['scale_bbox'] = scale_bbox
    result: ImagePipelineData = compose_node(data)

    tmpdir = str(tmpdir)
    result_image = result.processed_image
    print(result_image.shape)
    h, w, c = result_image.shape
    assert h % MUST_DIVIDED == 0
    assert w % MUST_DIVIDED == 0
    assert c == 3

    cv2.imwrite(os.path.join(tmpdir, img_name), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))