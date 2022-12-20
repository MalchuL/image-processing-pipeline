# Image processing pipeline

Modular image processing pipeline using OpenCV and Python generators. Inspired by Albumentations framework to process
images

## Setup environment

This project is using [Conda](https://conda.io) for project environment management.

Setup the project environment:

    $ virtualenv -p python3.8 venv
    $ source venv/bin/activate

## Getting started

For stylization pipeline:

1. Create `config.yaml` with content:

```
_target_: pipeline.compose.Compose
nodes:
  - _target_: pipeline.readers.opencv_read_image.ReadOpenCVImage
  - _target_: pipeline.preprocess.resize2division.Resize2Dividable
    must_divided: &must_divided 32
  - _target_: pipeline.detectors.face_cropping.FaceCropping
    detector:
      _target_: pipeline.detectors.lib.mediapipe_detector.StatMediaPipeDetector
      target_size: &target_shape 256
      must_divide: *must_divided
  - _target_: pipeline.stylization.image_gan_stylization.ImageGANStylization
    inference_engine: &inference_engine
      _target_: pipeline.stylization.inference_engine.inference_engine_cache.get_onnx_inference
      model_path: # path to your onnx model
  - _target_: pipeline.stylization.bbox_gan_stylization.BBoxGANStylization
    inference_engine: *inference_engine
  - _target_: pipeline.postprocess.merging_crops.seamless_merging_crops.SeamlessMergingCrops
    crops_size: *target_shape
```

2. In your python script use next code

```

import logging
import os
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from pipeline.pipeline import Pipeline
from pipeline.processed_data import ImagePipelineData
from pipeline.utils.pipeline_data import get_data
from tests.pipeline import CONFIG
from utils.instantiate import instantiate


config_path = Path(datadir) / 'config.yaml'

config = OmegaConf.load(config_path)
pipeline: Pipeline = instantiate(config)


data = get_data(im_path)
result: ImagePipelineData = pipeline(data)
result_image = result.processed_image

cv2.imwrite(os.path.join(out_dir, img_name), cv2.cvtColor(result.processed_image, cv2.COLOR_RGB2BGR))
```

## Tests

`pytest` is used as a test framework. All tests are stored in `tests` folder. Run the tests:

```bash
$ pytest
```

## Resources and Credits

* For Unix like pipeline idea credits goes to this [Gist](https://gist.github.com/alexmacedo/1552724)
* Source of the example images and videos is [pixbay](https://pixabay.com)
* Some ideas and code snippets are borrowed from [pyimagesearch](https://www.pyimagesearch.com/)
* Color constants
  from [Python Color Constants Module](https://www.webucator.com/blog/2015/03/python-color-constants-module/)

## License

[MIT License](LICENSE)