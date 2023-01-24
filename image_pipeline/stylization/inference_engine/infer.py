from abc import ABC, abstractmethod

import numpy as np


class InferenceEngine(ABC):

    @abstractmethod
    def forward(self, img: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, img: np.ndarray):
        return self.forward(img)