from dataclasses import dataclass
from typing import Union, Iterable, List, Dict

import numpy as np


@dataclass()
class DetectionResult:
    bbox: Union[List, np.ndarray]
    crop: Union[np.ndarray, None] = None


@dataclass
class ImagePipelineData:
    """Class for keeping track of an item in inventory."""
    input_image_path: Union[str, None] = None
    input_image: np.ndarray = None
    detection_bboxes: List[DetectionResult] = None
    processed_detection_bboxes: List[DetectionResult] = None
    processed_image: np.ndarray = None
    output_path: Union[str, None] = None
