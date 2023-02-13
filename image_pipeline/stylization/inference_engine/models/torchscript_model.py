import logging

import numpy as np
import torch

try:
    import onnxruntime
except ImportError:
    logging.error('torch cannot be loaded. did you try use `pip install torch`. TorchScriptModel cannot be used')

class TorchScriptModel:
    def __init__(self, torchscrtpt_path, use_cuda=False):
        self.use_cuda = use_cuda
        self.path = torchscrtpt_path
        self.scripted_module = torch.jit.load(torchscrtpt_path)
        if self.use_cuda:
            self.scripted_module.cuda()

    def __call__(self, img):
        img = torch.from_numpy(img.astype(np.float32))
        with torch.inference_mode():
            if self.use_cuda:
                img = img.cuda()
            outs = self.scripted_module(img)
            if self.use_cuda:
                outs = outs.cpu()
            outs = outs.numpy()
        return outs
