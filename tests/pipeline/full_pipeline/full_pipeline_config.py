import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from pipeline.pipeline import Pipeline
from pipeline.processed_data import ImagePipelineData
from pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG
from utils.instantiate import instantiate


@pytest.mark.skip('TODO add normal path to model')
@pytest.mark.parametrize('img_name', CONFIG.IMAGES + CONFIG.IMAGES_NO_FACES)
def test_full_pipeline(datadir, img_name, tmpdir):
    print('curdir', os.path.curdir)

    im_path = Path(datadir) / CONFIG.IM_FOLDER / img_name
    config_path = Path(datadir) / CONFIG.CONFIGS_FOLDER / 'config.yaml'

    config = OmegaConf.load(config_path)
    pipeline: Pipeline = instantiate(config)

    compose_node = pipeline

    data = get_data(im_path)
    result: ImagePipelineData = compose_node(data)
    tmpdir = str(tmpdir)
    result_image = result.processed_image
    print(result_image.shape)
    h, w, c = result_image.shape
    print(result.processed_detection_bboxes)

    assert c == 3

    cv2.imwrite(os.path.join(tmpdir, img_name), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))
