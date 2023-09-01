from hzhu_util import hutil

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from matplotlib import pyplot as plt
from typing import Union, Iterable, Tuple
from functools import cached_property
from collections import abc

HNumericType = Union[int, float, bool, np.number]
HArrayType = Union[Iterable, np.ndarray, torch.Tensor, Image.Image]


class HArray(np.ndarray):
    def __new__(cls, src: HArrayType, deep_copy: bool = False, **kwargs):
        hutil.check_type(deep_copy, bool)
        hutil.check_type(src, (abc.Iterable, np.ndarray, torch.Tensor, Image.Image))
        if deep_copy:
            obj = np.array(src).view(cls)
        else:
            obj = np.asanyarray(src).view(cls)
        obj._new_attr_dict = kwargs
        for key, value in obj._new_attr_dict.items():
            if hasattr(obj, key):
                raise ValueError(f'attribute "{key}" already exits.')
            setattr(obj, key, value)
        return obj

    def __array_finalize__(self, obj):
        # self the new instance, obj is the old instance
        if obj is None:
            return

        if hasattr(obj, "_new_attr_dict") and not hasattr(self, "_new_attr_dict"):
            for key, value in obj._new_attr_dict.items():
                if hasattr(self, key):
                    raise ValueError(f"attribute {key} already exits.")
                setattr(self, key, value)
            self._new_attr_dict = obj._new_attr_dict


class HVec(HArray):
    @staticmethod
    def _check_shape(x):
        if len(x.shape) != 1:
            raise ValueError(x.shape)

    def __new__(
        cls,
        src: Union[Iterable, np.ndarray, torch.Tensor],
        deep_copy: bool = False,
        **kwargs,
    ):
        obj = super().__new__(cls, src=src, deep_copy=deep_copy, **kwargs)
        cls._check_shape(obj)
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.__class__._check_shape(self)


class HVec2(HVec):
    @staticmethod
    def _check_shape(x):
        if len(x.shape) != 1:
            raise ValueError(x.shape)
        if x.shape[0] != 2:
            raise ValueError(x.shape)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @x.setter
    def x(self, value: HNumericType):
        self[0] = value

    @y.setter
    def y(self, value: HNumericType):
        self[1] = value


class HVec3(HVec2):
    @staticmethod
    def _check_shape(x):
        if len(x.shape) != 1:
            raise ValueError(x.shape)
        if x.shape[0] != 3:
            raise ValueError(x.shape)

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value: HNumericType):
        self[2] = value


class HVec4(HVec3):
    @staticmethod
    def _check_shape(x):
        if len(x.shape) != 1:
            raise ValueError(x.shape)
        if x.shape[0] != 4:
            raise ValueError(x.shape)

    @property
    def w(self):
        return self[3]

    @w.setter
    def w(self, value: HNumericType):
        self[3] = value


class HImage(HArray):
    @staticmethod
    def _check_shape(x):
        if not x.shape:
            return
        if len(x.shape) <= 1 or len(x.shape) > 3:
            raise ValueError(x.shape)
        if x.shape[0] <= 4:
            raise TypeError("Channel last image expected.", x.shape)
        if len(x.shape) == 3 and (x.shape[-1] == 2 or x.shape[-1] > 4):
            raise ValueError(x.shape)

    def __new__(
        cls,
        src: Union[Iterable, np.ndarray, torch.Tensor],
        deep_copy: bool = False,
        **kwargs,
    ):
        obj = super().__new__(
            cls, src=src, deep_copy=deep_copy, _deep_copy=deep_copy, **kwargs
        )
        cls._check_shape(obj)
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.__class__._check_shape(self)

    @classmethod
    def from_path(cls, path: str):
        hutil.check_type(path, str)
        return cls.__new__(cls, src=Image.open(path), deep_copy=True)

    def plot(self, axis_off=True, **kwargs):
        plt.imshow(self, **kwargs)
        if axis_off:
            plt.axis("off")

    @property
    def C(self) -> int:
        return 1 if len(self.shape) == 2 else self.shape[2]

    @property
    def H(self) -> int:
        return self.shape[0]

    @property
    def W(self) -> int:
        return self.shape[1]

    def to_torch(self, channel_first=True, deep_copy=True):
        result = torch.tensor(self) if deep_copy else torch.from_numpy(self)
        if channel_first:
            result = result.permute(2, 0, 1)
        return result

    def __str__(self):
        return f"{self.__class__}, {self.shape=}, {self._deep_copy=}"

    __repr__ = __str__
