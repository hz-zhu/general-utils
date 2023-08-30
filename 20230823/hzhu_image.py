from hzhu_util import hutil

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from matplotlib import pyplot as plt

class HImage(np.ndarray):

    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, **kwargs):
        obj = super().__new__(cls, shape, dtype,
                              buffer, offset, strides, order)
        for key, value in kwargs.items():
            setattr(obj, key, value)
        return obj

    def check_shape(self):
        if len(self.shape)>3 or len(self.shape)<2: raise ValueError(self.shape)

    def __array_finalize__(self, obj):
        return
    
    def cls_decorator(method, *margs, **mkwargs):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                method(self, *margs, **mkwargs)
                return func(self, *args, **kwargs)
            return wrapper
        return decorator

    @classmethod
    def from_path(cls, path: str):
        hutil.check_type(path, str)
        img_pil = Image.open(path)
        x = np.array(img_pil)
        return cls(shape=x.shape, dtype=x.dtype, buffer=x.tobytes(), deep_copy=True)
    
    @classmethod
    def from_numpy(cls, src: np.ndarray, deep_copy: bool = False):
        hutil.check_type(src, np.ndarray)
        hutil.check_type(deep_copy, bool)

        src = src.copy() if deep_copy else src
        return cls(shape=src.shape, dtype=src.dtype, buffer=src.tobytes(), deep_copy=deep_copy)
        
    @classmethod
    def from_torch(cls, src: torch.Tensor, deep_copy: bool = False):
        hutil.check_type(src, torch.Tensor)
        hutil.check_type(deep_copy, bool)

        src = src.numpy().copy() if deep_copy else src.numpy()
        return cls(shape=src.shape, dtype=src.dtype, buffer=src.tobytes(), deep_copy=deep_copy)
    
    @cls_decorator(method=check_shape)
    def plot(self, axis_off: bool = True):
        hutil.check_type(axis_off, bool)
        plt.imshow(self)
        if axis_off: plt.axis('off')

    @cls_decorator(method=check_shape)
    def to_tensor(self, deep_copy: bool = False) -> torch.Tensor:
        return torch.tensor(self) if deep_copy else torch.from_numpy(self)