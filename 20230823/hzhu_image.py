from hzhu_util import hutil

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image

@dataclass
class HImage(np.ndarray):

    def validataion(self):
        if len(self.shape)>3 or len(self.shape)<2: raise ValueError(self.shape)

    def __array_finalize__(self, obj):
        self.validataion

    @classmethod
    def from_path(cls, path: str):
        hutil.check_type(path, str)
        img_pil = Image.open(path)
        x = np.array(img_pil)
        return cls(shape=x.shape, dtype=x.dtype, buffer=x.tobytes())