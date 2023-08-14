from typing import Tuple, List, Iterable, Callable, Union, Any, ClassVar
import os
import warnings
from dataclasses import dataclass
import uuid
from datetime import datetime
from collections import abc


@dataclass(frozen=True)
class HUtil:
    ID: ClassVar[uuid.UUID] = uuid.uuid1()
    now: ClassVar[datetime] = datetime.now()
    cwd: ClassVar[str] = os.getcwd()

    @property
    def ID_str(self):
        return str(self.ID)

    @property
    def now_str(self):
        return self.now.strftime("%m/%d/%Y, %H:%M:%S")

    def __post_init__(self):
        HUtil.check_type(self.ID, uuid.UUID)
        HUtil.check_type(self.now, datetime)
        HUtil.check_type(self.cwd, str)

    @staticmethod
    def is_iterable(x: Any) -> bool:
        return isinstance(x, abc.Iterable)

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


hutil = HUtil()
