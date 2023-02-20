from dataclasses import dataclass
from typing import Union, Iterable, List, Dict, Optional

import numpy as np


@dataclass
class DetectionResult:
    bbox: Union[List, np.ndarray]
    crop: Union[np.ndarray, None] = None

@dataclass
class AdditionalArgs:
    detection: Optional[Dict] = None

@dataclass
class ImagePipelineData:
    """Class for keeping track of an item in inventory."""
    input_image_path: Optional[str] = None
    input_image: np.ndarray = None
    detection_bboxes: List[DetectionResult] = None
    processed_detection_bboxes: List[DetectionResult] = None
    processed_image: np.ndarray = None
    output_path: Optional[str] = None

    additional_kwargs: Optional[AdditionalArgs] = AdditionalArgs({})
