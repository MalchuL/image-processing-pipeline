import os.path
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.compose import Compose
from pipeline.detectors.face_cropping import FaceCropping
from pipeline.detectors.lib.mediapipe_detector import StatMediaPipeDetector
from pipeline.preprocess.resize2division import Resize2Dividable
from pipeline.processed_data import ImagePipelineData
from pipeline.readers.opencv_read_image import ReadOpenCVImage
from pipeline.stylization.bbox_gan_stylization import BBoxGANStylization
from pipeline.stylization.image_gan_stylization import ImageGANStylization
from pipeline.stylization.inference_engine.onnx_image_infer import ONNXImageInference
from pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG

MUST_DIVIDED = 32
TARGET_SHAPE = 256


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
def test_stylization_pipeline(datadir, img_name, tmpdir):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    read_node = ReadOpenCVImage()
    resizer_node = Resize2Dividable(must_divided=MUST_DIVIDED)
    model_path = Path(datadir) / CONFIG.ONNX_MODEL_PATH
    inference_engine = ONNXImageInference(model_path)
    stylization_node = ImageGANStylization(inference_engine)
    compose_node = Compose([read_node, resizer_node, stylization_node])

    data = get_data(im_path)
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
def test_cropping_stylization_pipeline(datadir, img_name, tmpdir):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    read_node = ReadOpenCVImage()
    resizer_node = Resize2Dividable(must_divided=MUST_DIVIDED)
    detector = StatMediaPipeDetector(TARGET_SHAPE)
    cropping_node = FaceCropping(detector)
    model_path = Path(datadir) / CONFIG.ONNX_MODEL_PATH
    inference_engine = ONNXImageInference(model_path)
    image_stylization_node = ImageGANStylization(inference_engine)
    crops_stylization_node = BBoxGANStylization(inference_engine)
    compose_node = Compose([read_node, resizer_node, cropping_node, image_stylization_node, crops_stylization_node])

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    tmpdir = str(tmpdir)
    result_image = result.processed_image
    print(result_image.shape)
    h, w, c = result_image.shape
    assert h % MUST_DIVIDED == 0
    assert w % MUST_DIVIDED == 0
    assert c == 3

    cv2.imwrite(os.path.join(tmpdir, img_name), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))
    assert len(result.processed_detection_bboxes) > 0
    for i, (det_boxes, proc_boxes) in enumerate(zip(result.detection_bboxes, result.processed_detection_bboxes)):
        assert np.any(det_boxes.crop != proc_boxes.crop)
        assert np.all(det_boxes.bbox == proc_boxes.bbox)
        crop = proc_boxes.crop
        h, w, c = crop.shape
        assert h == w == TARGET_SHAPE
        cv2.imwrite(os.path.join(tmpdir, f'{i}_style_{img_name}'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(tmpdir, f'{i}_base_{img_name}'), cv2.cvtColor(det_boxes.crop, cv2.COLOR_RGB2BGR))


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
def test_no_changing_original_stylization_pipeline(datadir, img_name, tmpdir):
    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    read_node = ReadOpenCVImage()
    resizer_node = Resize2Dividable(must_divided=MUST_DIVIDED)
    detector = StatMediaPipeDetector(TARGET_SHAPE)
    cropping_node = FaceCropping(detector)
    model_path = Path(datadir) / CONFIG.ONNX_MODEL_PATH
    inference_engine = ONNXImageInference(model_path)
    crops_stylization_node = BBoxGANStylization(inference_engine)
    compose_node = Compose([read_node, resizer_node, cropping_node, crops_stylization_node])

    idt_pipeline = Compose([read_node, resizer_node])

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)

    data = get_data(im_path)
    result_idt: ImagePipelineData = idt_pipeline(data)

    tmpdir = str(tmpdir)
    result_image = result.processed_image
    print(result_image.shape)
    h, w, c = result_image.shape
    assert h % MUST_DIVIDED == 0
    assert w % MUST_DIVIDED == 0
    assert c == 3

    assert np.all(result_idt.processed_image == result.processed_image)

    cv2.imwrite(os.path.join(tmpdir, img_name), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))
    assert len(result.processed_detection_bboxes) > 0
    for i, (det_boxes, proc_boxes) in enumerate(zip(result.detection_bboxes, result.processed_detection_bboxes)):
        assert np.any(det_boxes.crop != proc_boxes.crop)
        assert np.all(det_boxes.bbox == proc_boxes.bbox)
        crop = proc_boxes.crop
        h, w, c = crop.shape
        assert h == w == TARGET_SHAPE
        cv2.imwrite(os.path.join(tmpdir, f'{i}_style_{img_name}'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(tmpdir, f'{i}_base_{img_name}'), cv2.cvtColor(det_boxes.crop, cv2.COLOR_RGB2BGR))