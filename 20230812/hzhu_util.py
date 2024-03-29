from typing import Tuple, List, Iterable, Callable, Union, Any, ClassVar
import os
import warnings
from dataclasses import dataclass, field, fields
import uuid
import re
import time
from datetime import datetime
from collections import abc, Counter
from functools import cached_property, partial
import shutil
import json


@dataclass(frozen=True)
class HUtil:
    # Only one unique instance can be crated. Duplicate instances share the same fields
    ID: ClassVar[uuid.UUID] = uuid.uuid1()
    now: ClassVar[datetime] = datetime.now()
    cwd: ClassVar[str] = os.getcwd()
    timers: ClassVar[List[int]] = list()  # Time in nanosecond since epoch

    @cached_property
    def ID_str(self) -> str:
        return str(self.ID)

    @cached_property
    def now_str(self) -> str:
        return self.now.strftime("%m/%d/%Y %H:%M:%S")

    @property
    def time(self) -> float:
        current_time = time.time_ns()
        self.timers.append(current_time)
        return (current_time-self.timers[-2])/10e9

    def __post_init__(self):
        HUtil.check_type(self.ID, uuid.UUID)
        HUtil.check_type(self.now, datetime)
        HUtil.check_type(self.cwd, str)
        self.timers.append(time.time_ns())
        HUtil.check_iterable_type(self.timers, int)

    @staticmethod
    def is_iterable(x: Any) -> bool:
        return isinstance(x, abc.Iterable)

    @staticmethod
    def is_callable(x: Any) -> bool:
        return isinstance(x, abc.Callable)

    @staticmethod
    def check_type(x: Any, dtype: Union[type, Tuple[type]], raise_error: bool = True, self_check: bool = True) -> bool:
        if self_check:
            HUtil.check_type(raise_error, bool,
                             raise_error=True, self_check=False)
            HUtil.check_type(self_check, bool,
                             raise_error=True, self_check=False)
            if HUtil.is_iterable(dtype):
                HUtil.check_type(
                    dtype, tuple, raise_error=True, self_check=False)
                for item in dtype:
                    HUtil.check_type(
                        item, type, raise_error=True, self_check=False)
            else:
                HUtil.check_type(
                    dtype, type, raise_error=True, self_check=False)

        if not isinstance(x, dtype):
            if HUtil.is_iterable(dtype):
                msg = f"Expecting object of either types {[item.__name__ for item in dtype]} yet get type [{x.__class__.__name__}]"
            else:
                msg = f'Expecting object {x} of type "{dtype.__name__}" yet get type "{x.__class__.__name__}"'
            if raise_error:
                raise TypeError(msg)
            else:
                warnings.warn(msg)

            return False
        return True

    @staticmethod
    def check_iterable_type(x: Iterable, dtype: Union[type, Tuple[type]], raise_error: bool = True, self_check: bool = True) -> bool:
        if self_check:
            HUtil.check_type(
                x, abc.Iterable, raise_error=True, self_check=False)
        for i, item in enumerate(x):
            if not HUtil.check_type(item, dtype=dtype, raise_error=raise_error, self_check=False if i else True):
                return False
        return True

    @staticmethod
    def str_contain_keyword(s: str, keyword: str) -> bool:
        HUtil.check_type(s, str)
        HUtil.check_type(keyword, str)

        return keyword in s

    @staticmethod
    def str_contain_any(s: str, keywords: Iterable[str]) -> bool:
        HUtil.check_type(s, str)
        HUtil.check_type(keywords, abc.Iterable)
        HUtil.check_iterable_type(keywords, str)

        return any([HUtil.str_contain_keyword(s, item) for item in keywords])

    @staticmethod
    def str_contain_all(s: str, keywords: Iterable[str]) -> bool:
        HUtil.check_type(s, str)
        HUtil.check_type(keywords, abc.Iterable)
        HUtil.check_iterable_type(keywords, str)

        return all([HUtil.str_contain_keyword(s, item) for item in keywords])

    @staticmethod
    def str_contains(s: str, contain_all: Iterable[str] = '', contain_any: Iterable[str] = '', contain_none: Iterable[str] = '') -> bool:
        if isinstance(contain_all, str):
            contain_all = [contain_all]
        if isinstance(contain_any, str):
            contain_any = [contain_any]
        if isinstance(contain_none, str):
            contain_none = [contain_none]

        HUtil.check_iterable_type(contain_all, str)
        HUtil.check_iterable_type(contain_any, str)
        HUtil.check_iterable_type(contain_none, str)

        contain_all_flag = HUtil.str_contain_all(s, contain_all)
        contain_any_flag = HUtil.str_contain_any(s, contain_any)
        if len(contain_none) == 1 and not contain_none[0]:
            contain_none_flag = True
        else:
            contain_none_flag = (not HUtil.str_contain_any(s, contain_none))

        return contain_all_flag and contain_any_flag and contain_none_flag

    @staticmethod
    def ls(path: str = os.getcwd(), full_dir=False, sort_key=lambda x: (HUtil.extract_numbers(x), x))  -> List[str]:
        HUtil.check_type(path, str)
        HUtil.check_type(full_dir, bool)
        if sort_key:
            HUtil.is_callable(sort_key)
        result = os.listdir(path)
        if sort_key:
            result = sorted(result, key=sort_key)
        return [os.path.join(path, item) for item in result] if full_dir else result

    @staticmethod
    def ls_file(path: str = os.getcwd(), full_dir=False, sort_key=lambda x: (HUtil.extract_numbers(x), x)) -> List[str]:
        HUtil.check_type(path, str)
        HUtil.check_type(full_dir, bool)
        if sort_key:
            HUtil.is_callable(sort_key)
        result = HUtil.ls(path=path, full_dir=False, sort_key=sort_key)
        result_full = [os.path.join(path, item) for item in result]
        return [rf if full_dir else r for r, rf in zip(result, result_full) if os.path.isdir(rf)]

    @staticmethod
    def ls_dir(path: str = os.getcwd(), full_dir=False, sort_key=lambda x: (HUtil.extract_numbers(x), x)) -> List[str]:
        HUtil.check_type(path, str)
        HUtil.check_type(full_dir, bool)
        if sort_key:
            HUtil.is_callable(sort_key)
        result = HUtil.ls(path=path, full_dir=False, sort_key=sort_key)
        result_full = [os.path.join(path, item) for item in result]
        return [rf if full_dir else r for r, rf in zip(result, result_full) if os.path.isdir(rf)]

    @staticmethod
    def extract_numbers(input_string) -> List[Union[int, float]]:
        HUtil.check_type(input_string, str)
        # Regular expression for matching numbers
        number_pattern = r"-?[\d]*\.?\d+(?:[eE][-+]?\d+)?"
        numbers = re.findall(number_pattern, input_string)
        numbers = [float(number) if "." in number or "e" in number.lower() else int(
            number) for number in numbers]
        return numbers

    @staticmethod
    def ls_name(path: str = os.getcwd(), *, full_dir=False, sort_key=lambda x: (HUtil.extract_numbers(x), x),
        contain_all: Iterable[str] = '', contain_any: Iterable[str] = '', contain_none: Iterable[str] = '') -> List[str]:
        HUtil.check_type(path, str)
        HUtil.check_type(full_dir, bool)
        if sort_key:
            HUtil.is_callable(sort_key)

        filter_func = partial(HUtil.str_contains, contain_all=contain_all,
                              contain_any=contain_any, contain_none=contain_none)
        result = list(filter(filter_func, HUtil.ls_file(
            path=path, full_dir=False, sort_key=sort_key)))
        result_full = [os.path.join(path, item) for item in result]

        return [rf if full_dir else r for r, rf in zip(result, result_full) if os.path.isdir(rf)]
    
    @staticmethod
    def mkdir(*args, exist_warning: bool=True) -> str:
        HUtil.check_iterable_type(args, str)
        HUtil.check_type(exist_warning, bool)

        path = os.path.join(*args)
        if os.path.exists(path):
            if exist_warning: warnings.warn(f"Folder {path} already exists.")
            return path
        else:
            os.makedirs(path)
            return path

    @staticmethod
    def newdir(*args, force_delete: bool=False, raise_error: bool=True) -> str:
        HUtil.check_iterable_type(args, str)
        HUtil.check_type(raise_error, bool)
        HUtil.check_type(force_delete, bool)

        path = os.path.join(*args)
        if os.path.exists(path):
            msg = f"Folder {path} already exists."
            if raise_error: raise FileExistsError(msg)
            else:
                warnings.warn(msg)
                if force_delete:
                    shutil.rmtree(path)
                    HUtil.mkdir(path, exist_warning=True)
                    return path
                else:
                    return ''
        else:
            HUtil.mkdir(path, exist_warning=False)
            return path
        
    @staticmethod
    def get_struct(x: Any) -> Union[dict, str]:
        if isinstance(x, abc.Mapping):
            result = dict()
            for key, value in x.items():
                recurse = HUtil.get_struct(value)
                if recurse:
                    result[f"{key} <{value.__class__.__name__}>"] = recurse
                else:
                    result[key] = value.__class__.__name__
            return result
        elif isinstance(x, str):
            return x.__class__.__name__
        elif isinstance(x, abc.Iterable):
            res_hashable = []
            res_unhashable = []
            for item in x:
                recurse = HUtil.get_struct(item)
                if recurse:
                    try:
                        hash(recurse)
                        res_hashable.append(recurse)
                    except:
                        res_unhashable.append(recurse)
            result = []
            if res_hashable: result.append(Counter(res_hashable))
            if res_unhashable: result.append(res_unhashable)
            return result

        else:
            return x.__class__.__name__
        
    @staticmethod
    def disp_struct(x: Any) -> None:
        res = json.dumps(HUtil.get_struct(x), indent=4)
        print(res)

hutil = HUtil()
