from hzhu_util import hutil

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from matplotlib import pyplot as plt
from typing import Union, Iterable
from functools import cached_property
from collections import abc


class HVec(np.ndarray):
    # A vector of fixed length
    def __new__(cls, length: int, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        hutil.check_type(length, int)
        obj = super().__new__(cls, length, dtype,
                              buffer, offset, strides, order)
        obj._fixed_len = length
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self._fixed_len = getattr(obj, '_fixed_len', 0)

    def cls_decorator(method, *margs, **mkwargs):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                method(self, *margs, **mkwargs)
                return func(self, *args, **kwargs)
            return wrapper
        return decorator

    def check_shape(self) -> None:
        if len(self.shape) != 1:
            raise ValueError(self.shape)
        if self.shape[0] != self._fixed_len:
            raise ValueError(self.shape, self._fixed_len)

    @classmethod
    def from_numbers(cls, *args, dtype: type = float):
        hutil.check_type(dtype, type)
        length = len(args)
        data_list = [dtype(item) for item in args]
        x = np.array(data_list)
        return cls(length=length, dtype=x.dtype, buffer=x.tobytes())

    @classmethod
    def from_iterable(cls, x: Iterable, dtype: type = float):
        hutil.check_type(dtype, type)
        hutil.check_iterable(x)
        return cls.from_numbers(*x, dtype=dtype)


class HVec2(HVec):

    def check_shape(self) -> None:
        if self._fixed_len != 2:
            raise ValueError(self.shape)
        return super().check_shape()

    @property
    @HVec.cls_decorator(check_shape)
    def x(self): 
        return self[0]

    @property
    @HVec.cls_decorator(check_shape)
    def y(self): 
        return self[1]
    
class HVec3(HVec2):

    def check_shape(self) -> None:
        if self._fixed_len != 3:
            raise ValueError(self.shape)
        return super().check_shape()

    @property
    @HVec.cls_decorator(check_shape)
    def z(self): 
        return self[2]
    
class HVec4(HVec2):

    def check_shape(self) -> None:
        if self._fixed_len != 4:
            raise ValueError(self.shape)
        return super().check_shape()

    @property
    @HVec.cls_decorator(check_shape)
    def w(self): 
        return self[3]


class HImage(np.ndarray):

    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, **kwargs):
        hutil.check_iterable(shape)
        hutil.check_iterable_type(shape, int)
        obj = super().__new__(cls, shape, dtype,
                              buffer, offset, strides, order)
        for key, value in kwargs.items():
            setattr(obj, key, value)
        if not HImage.is_channel_last_image(obj):
            raise ValueError(obj.shape)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.deep_copy = getattr(obj, 'deep_copy', False)

    def check_shape(self):
        if len(self.shape) > 3 or len(self.shape) < 2:
            raise ValueError(self.shape)
        if self.shape[0] < 5:
            raise ValueError(self.shape)

    def __str__(self) -> str:
        return f'<HImage> shape={self.shape}, dtype={self.dtype}, deep_copy={self.deep_copy}'

    __repr__ = __str__

    @property
    def C(self) -> int:
        return 1 if len(self.shape) == 2 else self.shape[2]

    @property
    def H(self) -> int:
        return self.shape[0]

    @property
    def W(self) -> int:
        return self.shape[1]

    @staticmethod
    def is_channel_last_image(x: Union[np.ndarray, torch.Tensor]) -> bool:
        hutil.check_type(x, (np.ndarray, torch.Tensor))
        if len(x.shape) > 3 or len(x.shape) < 2:
            return False
        if x.shape[0] < 5:
            return False
        return True

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
        if axis_off:
            plt.axis('off')

    @cls_decorator(method=check_shape)
    def to_tensor(self, *, deep_copy: bool = True, channel_first: bool = True) -> torch.Tensor:
        hutil.check_type(deep_copy, bool)
        hutil.check_type(channel_first, bool)
        if channel_first:
            if self.C == 1:
                return torch.tensor(self) if deep_copy else torch.from_numpy(self)
            else:
                return torch.tensor(self).permute(2, 0, 1) if deep_copy else torch.from_numpy(self).permute(2, 0, 1)
        else:
            return torch.tensor(self) if deep_copy else torch.from_numpy(self)


@dataclass(frozen=True, order=True)
class HBBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @cached_property
    def h(self) -> int:
        return self.y2-self.y1

    @cached_property
    def w(self) -> int:
        return self.x2-self.x1

    @cached_property
    def area(self) -> int:
        return self.h*self.w

    def __post_init__(self):
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError(self.x1, self.y1)
        if self.x1 >= self.x2:
            raise ValueError(self.x1, self.x2)
        if self.y1 >= self.y2:
            raise ValueError(self.y1, self.y2)


class HBoolMask(HImage):
    # Binary mask (H x W)
    def check_shape(self) -> None:
        if len(self.shape) != 2:
            raise ValueError(self.shape)
        if self.dtype != np.bool_:
            raise TypeError(self.dtype)

    @property
    def area(self) -> int:
        return int(self.sum())


class HIntMask(HImage):
    # Binary mask (H x W x C)
    def check_shape(self) -> None:
        if len(self.shape) != 3:
            raise ValueError(self.shape)
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError(self.dtype)
