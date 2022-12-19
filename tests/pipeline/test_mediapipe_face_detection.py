import os.path
from pathlib import Path

import cv2
import pytest

from pipeline.compose import Compose
from pipeline.detectors.face_cropping import FaceCropping
from pipeline.detectors.lib.mediapipe_detector import StatMediaPipeDetector
from pipeline.processed_data import ImagePipelineData
from pipeline.readers.opencv_read_image import ReadOpenCVImage
from pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG

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
        assert det_boxes.bbox[2] < w
        assert det_boxes.bbox[3] < h
        assert det_boxes.bbox[0] < det_boxes.bbox[2]
        assert det_boxes.bbox[1] < det_boxes.bbox[3]

        cv2.imwrite(os.path.join(tmpdir, str(i) + '_' + img_name), cv2.cvtColor(det_boxes.crop, cv2.COLOR_RGB2BGR))
