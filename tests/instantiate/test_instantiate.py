from pathlib import Path

import pytest

from utils.instantiate import instantiate
from omegaconf import OmegaConf

CONFIGS_FOLDER = 'configs'
def test_files(datadir):
    files = Path(datadir) / CONFIGS_FOLDER
    print(files)
    for file in files.glob('*'):
        dict = OmegaConf.load(file)
        res = instantiate(dict)
        print(res)
    pass