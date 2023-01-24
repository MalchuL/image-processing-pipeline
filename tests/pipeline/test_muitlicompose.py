from pathlib import Path

import pytest

from image_pipeline.compose import Compose
from image_pipeline.detectors.face_cropping import FaceCropping
from image_pipeline.detectors.lib.mediapipe_detector import StatMediaPipeDetector
from image_pipeline.preprocess.resize2division import Resize2Dividable
from image_pipeline.processed_data import ImagePipelineData
from image_pipeline.readers.opencv_read_image import ReadOpenCVImage
from image_pipeline.stylization.bbox_gan_stylization import BBoxGANStylization
from image_pipeline.stylization.inference_engine.onnx_image_infer import ONNXImageInference
from image_pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG

MUST_DIVIDED = 16


@pytest.mark.parametrize('img_name', CONFIG.IMAGES)
@pytest.mark.parametrize("must_divided1", [18, 14])
def test_multi_resize_node(datadir, img_name, must_divided1):
    read_node = ReadOpenCVImage()
    resize_node_1 = Resize2Dividable(must_divided=must_divided1)
    resize_node_2 = Resize2Dividable(must_divided=MUST_DIVIDED)
    detector = StatMediaPipeDetector(256)
    cropping_node = FaceCropping(detector)
    model_path = Path(datadir) / CONFIG.ONNX_MODEL_PATH
    inference_engine = ONNXImageInference(model_path)
    crops_stylization_node = BBoxGANStylization(inference_engine)
    compose_node = Compose([read_node, Compose([resize_node_1, cropping_node, crops_stylization_node]), resize_node_2])

    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    h, w, c = result.processed_image.shape
    assert h % MUST_DIVIDED == w % MUST_DIVIDED == 0
