from functools import lru_cache as cache

from image_pipeline.stylization.inference_engine.onnx_image_infer import ONNXImageInference
from image_pipeline.stylization.inference_engine.torchscript_image_infer import TorchScriptImageInference


@cache
def get_onnx_inference(model_path, chunk_size=4):
    inference_engine = ONNXImageInference(model_path, chunk_size=chunk_size)
    return inference_engine


@cache
def get_torchscript_inference(model_path, use_gpu=False, chunk_size=4):
    inference_engine = TorchScriptImageInference(model_path, use_gpu=use_gpu, chunk_size=chunk_size)
    return inference_engine
