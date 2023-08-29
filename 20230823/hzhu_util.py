from typing import Tuple, List, Iterable, Callable, Union, Any, ClassVar
import os
import warnings
from dataclasses import dataclass, is_dataclass, asdict
import uuid
import re
import time
from datetime import datetime
from collections import abc, Counter, UserList, UserDict
from functools import cached_property, partial
import shutil


@dataclass(frozen=True)
class HUtil:
    # Only one unique instance can be crated. Duplicate instances share the same fields
    ID: ClassVar[uuid.UUID] = uuid.uuid1()
    start_time: ClassVar[datetime] = datetime.now()
    cwd: ClassVar[str] = os.getcwd()
    timers: ClassVar[List[int]] = list()  # Time in nanosecond since epoch
    # Debug is enabled by default, when enabled, hutil.disp() will print to screen
    debug_: ClassVar[list] = [None]

    def disp(self, *args, **kwargs) -> None:
        if self.debug:
            for item in args:
                print(item)
            for key, value in kwargs.items():
                print(f"{key} = {value}")

    def __str__(self) -> str:
        res = 'hzhu_util::hutil <HUtil (dataclass)>\n'
        res += f"  {self.ID = }\n"
        res += f"  {self.start_time = }\n"
        res += f"  {self.cwd = }\n"
        res += f"  {self.debug = }\n"
        return res

    __repr__ = __str__

    @cached_property
    def ID_str(self) -> str:
        return str(self.ID)

    @cached_property
    def start_time_str(self) -> str:
        return self.start_time.strftime("%Y%m%d|%H-%M-%S")

    @property
    def debug(self) -> bool:
        return bool(self.debug_)

    @property
    def now(self) -> datetime:
        return datetime.now()

    @property
    def now_str(self, format="%m/%d/%Y %H:%M:%S") -> str:
        HUtil.check_type(format, str)
        return datetime.now().strftime(format)

    @property
    def timer(self) -> float:
        current_time = time.time_ns()
        self.timers.append(current_time)
        return (current_time-self.timers[-2])/10e9

    def enable_debug(self) -> None:
        self.debug_.append(None)

    def disable_debug(self) -> None:
        self.debug_.clear()

    def __post_init__(self):
        HUtil.check_type(self.ID, uuid.UUID)
        HUtil.check_type(self.start_time, datetime)
        HUtil.check_type(self.cwd, str)
        HUtil.check_type(self.debug_, list)
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
    def check_iterable(x: Any, raise_error: bool = True, self_check: bool = True) -> bool:
        return HUtil.check_type(x=x, dtype=abc.Iterable, raise_error=raise_error, self_check=self_check)

    @staticmethod
    def check_callable(x: Any, raise_error: bool = True, self_check: bool = True) -> bool:
        return HUtil.check_type(x=x, dtype=abc.Callable, raise_error=raise_error, self_check=self_check)

    @staticmethod
    def check_dataclass(x: Any, raise_error: bool = True, self_check: bool = True) -> bool:
        if self_check:
            HUtil.check_type(raise_error, bool,
                             raise_error=True, self_check=False)
            HUtil.check_type(self_check, bool,
                             raise_error=True, self_check=False)

        if not is_dataclass(x):
            msg = f'Expecting object {x} of type "dataclass" yet get type "{x.__class__.__name__}"'
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
    def ls(path: str = os.getcwd(), full_dir=False, sort_key=lambda x: (HUtil.extract_numbers(x), x)) -> List[str]:
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
    def mkdir(*args, exist_warning: bool = True) -> str:
        HUtil.check_iterable_type(args, str)
        HUtil.check_type(exist_warning, bool)

        path = os.path.join(*args)
        if os.path.exists(path):
            if exist_warning:
                warnings.warn(f"Folder {path} already exists.")
            return path
        else:
            os.makedirs(path)
            return path

    @staticmethod
    def newdir(*args, force_delete: bool = False, raise_error: bool = True) -> str:
        HUtil.check_iterable_type(args, str)
        HUtil.check_type(raise_error, bool)
        HUtil.check_type(force_delete, bool)

        path = os.path.join(*args)
        if os.path.exists(path):
            msg = f"Folder {path} already exists."
            if raise_error:
                raise FileExistsError(msg)
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
    def disp_struct_full(x: Any) -> None:
        def recurse(x, depth):
            if isinstance(x, str):
                return 'str'
            if is_dataclass(x):
                return recurse(asdict(x), depth=depth)
            elif isinstance(x, dict):
                pad = '  '*depth
                return '{\n'+pad +\
                    (';\n'+pad).join(
                        (recurse(key, depth=depth+1) +
                         ':'+recurse(value, depth=depth+1))
                        for key, value in x.items())\
                    + '\n'+pad+'}'
            elif isinstance(x, Iterable):
                return '['+','.join((recurse(item, depth=depth+1) for item in x))+']'
            else:
                return x.__class__.__name__
        hutil.disp(recurse(x, 0))

    @staticmethod
    def disp_struct(x: Any, sep: str = ':   ') -> None:
        hutil.check_type(sep, str)

        def recurse(x, depth):
            pad = sep*depth
            if isinstance(x, str):
                return f"<{x.__class__.__name__}>"
            if is_dataclass(x):
                return f"<{x.__class__.__name__}>\n{pad}"+recurse(asdict(x), depth=depth)
            elif isinstance(x, dict):
                return '{\n'+pad +\
                    (';\n'+pad).join(
                        (str(key)+' '+recurse(key, depth=depth+1) +
                         ': '+recurse(value, depth=depth+1))
                        for key, value in x.items())\
                    + '\n'+pad+'}'
            elif isinstance(x, Iterable):
                local_structs = [recurse(item, depth=depth+1) for item in x]
                local_counter = Counter(local_structs)
                return '[\n'+pad +\
                    (',\n'+pad).join('x'+str(value)+' '+key for key, value in local_counter.items())\
                    + '\n'+pad+']'
            else:
                return f"<{x.__class__.__name__}>"
        hutil.disp(recurse(x, 0))


hutil = HUtil()


class HDataList(UserList):

    def __init__(self, data: Union[Iterable, None] = None, dtype: Union[type, None] = None, force_type: bool = True):
        if data:
            hutil.check_iterable(data)
        self.dtype = dtype
        self.force_type = force_type
        if self.dtype:
            hutil.check_type(self.dtype, type)
        hutil.check_type(self.force_type, bool)

        super().__init__()
        self.extend(data)

    def append(self, item: Any) -> None:
        if self.force_type:
            if self:
                hutil.check_type(item, self.dtype)
            else:
                hutil.check_dataclass(item)
                self.dtype = item.__class__
        super().append(item)

    def __setitem__(self, idx: int, item: Any) -> None:
        hutil.check_type(idx, int)
        if self.force_type:
            if self:
                hutil.check_type(item, self.dtype)
            else:
                hutil.check_dataclass(item)
                self.dtype = item.__class__
        super().__setitem__(idx, item)

    def extend(self, other: Iterable) -> None:
        hutil.check_type(other, abc.Iterable)
        for item in other:
            if self.force_type:
                if self:
                    hutil.check_type(item, self.dtype)
                else:
                    self.dtype = item.__class__
        return super().extend(other)

    def to_expanded_list(self):
        return [asdict(item) for item in self]


class HDataDict(UserDict):

    def __init__(self, 
        data: Union[abc.Iterable, None] = None,
        dtype: Union[type, None] = None, 
        key_str: str = '',
        force_type: bool = True,
        no_replacement: bool = True):

        if data:
            hutil.check_type(data, abc.Iterable)
        self.dtype = dtype
        self.force_type = force_type
        self.key_str = key_str
        self.no_replacement = no_replacement
        if self.dtype:
            hutil.check_type(self.dtype, type)
        hutil.check_type(self.force_type, bool)
        hutil.check_type(self.key_str, str)
        hutil.check_type(self.no_replacement, bool)

        super().__init__()
        if data:
            for item in data: self.add(item)

    def add(self, item: Any) -> None:
        if self.force_type:
            if self:
                hutil.check_type(item, self.dtype)
            else:
                hutil.check_dataclass(item)
                self.dtype = item.__class__
        key = getattr(item, self.key_str) if self.key_str else len(self)
        self.__setitem__(key, item)

    def __setitem__(self, key, value) -> None:
        if key in self and self.no_replacement:
            raise ValueError(f"{key} already exists in HDataDict.")
        super().__setitem__(key, value)

    def to_list(self) -> list:
        return list(self.data.values())

    def to_data_list(self) -> HDataList:
        return HDataList(self.data.values())
    
    def to_expanded_list(self) -> list:
        return [asdict(value) for value in self.data.values()]
    
    def to_expanded_dict(self) -> dict:
        return {key: asdict(value) for key, value in self.data.items()}
