from hzhu_util import hutil

import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from matplotlib import pyplot as plt
from typing import Union, Iterable, Tuple
from functools import cached_property
from collections import abc


class HArray(np.ndarray):
    # A array with custom attributes
    def __new__(cls, src, deep_copy: bool = False, **kwargs):
        hutil.check_type(src, (np.ndarray, abc.Iterable, torch.Tensor))
        hutil.check_type(deep_copy, bool)
        if deep_copy:

        else:
            obj = np.asanyarray(src).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        if self.shape != obj.shape:
            raise ValueError("Dimensions of ImmutableArray cannot be changed.")


class HVec(HArray):
    # A vector of fixed length
    @staticmethod
    def add_new_attr(obj, attr_dict: dict):
        super().add_new_attr(obj, attr_dict)
        obj._new_attr_dict['_fixed_len'] = obj.shape[0]
        obj._fixed_len = obj.shape[0]

    def check_validity(self) -> None:
        print('chedking')
        if len(self.shape) != 1:
            raise ValueError(self.shape)
        if hasattr(self, '_fixed_len') and self.shape[0] != self._fixed_len:
            raise ValueError(self.shape, self._fixed_len)


    @classmethod
    def from_numbers(cls, *args, dtype: Union[float, int, bool, np.dtype] = float, **kwargs):
        hutil.check_type(dtype, (type, np.dtype))
        length = len(args)
        data_list = [dtype(item) for item in args]
        x = np.array(data_list)
        return cls(shape=(length,), dtype=x.dtype, buffer=x.tobytes(), _fixed_len=length, **kwargs)

    @classmethod
    def from_iterable(cls, x: Iterable, dtype: Union[float, int, bool, np.dtype] = float, **kwargs):
        hutil.check_type(dtype, (type, np.dtype))
        hutil.check_iterable(x)
        return cls.from_numbers(*x, dtype=dtype, **kwargs)
    
    @classmethod
    def from_numpy(cls, src: np.ndarray, deep_copy: False = False, **kwargs):
        hutil.check_type(src, np.ndarray)
        hutil.check_type(deep_copy, bool)
        if deep_copy:
            return cls(shape=src.shape, dtype=src.dtype, buffer=src.tobytes(), **kwargs).copy()
        else:
            return cls(shape=src.shape, dtype=src.dtype, buffer=src.tobytes(), _fixed_len=src.shape[0], **kwargs)
