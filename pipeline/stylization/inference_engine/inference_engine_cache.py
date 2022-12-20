from functools import lru_cache as cache

from pipeline.stylization.inference_engine.onnx_image_infer import ONNXImageInference


@cache
def get_onnx_inference(model_path):
    inference_engine = ONNXImageInference(model_path)
    return inference_engine
