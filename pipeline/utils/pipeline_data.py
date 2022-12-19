from pathlib import Path
from typing import Union

import numpy as np

from pipeline.processed_data import ImagePipelineData


def get_data(img_or_path: Union[str, Path, np.ndarray]) -> ImagePipelineData:
    if isinstance(img_or_path, (str, Path)):
        data = ImagePipelineData(input_image_path=str(img_or_path))
    elif isinstance(img_or_path, np.ndarray):
        data = ImagePipelineData(input_image=img_or_path, processed_image=img_or_path.copy())
    else:
        raise ValueError('img_or_path must be in [str, Path, np.ndarray] types')
    return data
