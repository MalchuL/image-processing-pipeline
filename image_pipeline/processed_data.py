from dataclasses import dataclass, field
from typing import Union, List, Dict, Optional

import numpy as np


@dataclass
class DetectionResult:
    bbox: Union[List, np.ndarray]
    crop: Union[np.ndarray, None] = None

@dataclass
class AdditionalArgs:
    detection: Optional[Dict] = field(default_factory=dict)  # Possible keys: `scale_bbox` for mediapipe_detector
    img_padding: Optional[Dict] = field(default_factory=dict)  # Possible keys: `skip_padding` for mediapipe_detector
    output_cropping: Optional[List[int]] = None  # List of ints to crop [top, left, bottom, right]

@dataclass
class ImagePipelineData:
    """Class for keeping track of an item in inventory."""
    input_image_path: Optional[str] = None
    input_image: np.ndarray = None
    detection_bboxes: List[DetectionResult] = None
    processed_detection_bboxes: List[DetectionResult] = None
    processed_image: np.ndarray = None
    output_path: Optional[str] = None

    additional_kwargs: Optional[AdditionalArgs] = field(default_factory=AdditionalArgs)
