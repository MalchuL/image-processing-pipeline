import numpy as np
import onnxruntime


class ONNXModel:
    def __init__(self, onnx_mode_path, providers=None):
        self.path = onnx_mode_path
        self.ort_session = onnxruntime.InferenceSession(str(self.path), providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name

    def __call__(self, img):
        ort_inputs = {self.input_name: img.astype(dtype=np.float32)}
        ort_outs = self.ort_session.run(None, ort_inputs)[0]
        return ort_outs