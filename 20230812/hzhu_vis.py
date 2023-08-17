from dataclasses import asdict, is_dataclass
from typing import Tuple, List, Iterable, Callable, Union, Any, ClassVar, Mapping
from collections import Counter
from hzhu_util import hutil

class HVis:

    @staticmethod
    def struct_full(x: Any) -> str:
        def recurse(x, depth):
            if isinstance(x, str): return 'str'
            if is_dataclass(x):
                return recurse(asdict(x), depth=depth)
            elif isinstance(x, dict):
                pad = '  '*depth
                return '{\n'+pad+\
                    (';\n'+pad).join(\
                    (recurse(key, depth=depth+1)+':'+recurse(value, depth=depth+1)) \
                    for key, value in x.items())\
                    +'\n'+pad+'}'
            elif isinstance(x, Iterable):
                return '['+','.join((recurse(item, depth=depth+1) for item in x))+']'
            else:
                return x.__class__.__name__
        return recurse(x, 0)
    
    @staticmethod
    def struct(x: Any, sep: str=':   ') -> str:
        hutil.check_type(sep, str)
        def recurse(x, depth):
            pad = sep*depth
            if isinstance(x, str): return f"<{x.__class__.__name__}>"
            if is_dataclass(x):
                return f"<{x.__class__.__name__}>\n{pad}"+recurse(asdict(x), depth=depth)
            elif isinstance(x, dict):
                return '{\n'+pad+\
                    (';\n'+pad).join(\
                    (str(key)+' '+recurse(key, depth=depth+1)+': '+recurse(value, depth=depth+1)) \
                    for key, value in x.items())\
                    +'\n'+pad+'}'
            elif isinstance(x, Iterable):
                local_structs = [recurse(item, depth=depth+1) for item in x]
                local_counter = Counter(local_structs)
                return '[\n'+pad+\
                    (',\n'+pad).join('x'+str(value)+' '+key for key, value in local_counter.items())\
                    +'\n'+pad+']'
            else:
                return f"<{x.__class__.__name__}>"
        return recurse(x, 0)
        
        
hvis = HVis()