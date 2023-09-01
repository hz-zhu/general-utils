from hzhu_util import hutil, HDataList
from hzhu_image import HImage, HMask

from dataclasses import dataclass, field
from functools import cached_property
from torchvision.ops import masks_to_boxes

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
        hutil.check_type(self.x1, int)
        hutil.check_type(self.y1, int)
        hutil.check_type(self.x2, int)
        hutil.check_type(self.y2, int)
        if self.x1 < 0 or self.y1 < 0:
            raise ValueError(self.x1, self.y1)
        if self.x1 >= self.x2:
            raise ValueError(self.x1, self.x2)
        if self.y1 >= self.y2:
            raise ValueError(self.y1, self.y2)

    @classmethod
    def from_hmask(cls, src: HMask):
        hutil.check_type(src, HMask)
        msk = src.to_torch(deep_copy=True)
        x1, y1, x2, y2 = masks_to_boxes(msk.unsqueeze(0))[0,:].tolist()
        return cls(int(x1), int(y1), int(x2), int(y2))
    
@dataclass
class HMaskAnno:
    mask: HMask
    name: str = field(init=True, default_factory=lambda: '')
    id: int = field(init=True, default_factory=lambda: 0)

    def __post_init__(self):
        self.bbox: HBBox = HBBox.from_hmask(self.mask)
        hutil.check_type(self.mask, HMask)
        hutil.check_type(self.bbox, HBBox)
        hutil.check_type(self.name, str)
        hutil.check_type(self.id, int)

    @property
    def H(self) -> int:
        return self.mask.H
    
    @property
    def W(self) -> int:
        return self.mask.W
    
    @property
    def area(self) -> int:
        return int(self.mask.sum())

@dataclass
class HImageAnno:
    image: HImage
    annos: HDataList[HMaskAnno] = field(init=True, default_factory=lambda: HDataList(None, dtype=HMaskAnno))

    @property
    def H(self):
        return self.image.H
    
    @property
    def W(self):
        return self.image.W

    def _check(self, idx: int):
        hutil.check_type(idx, int)
        anno = self.annos[idx]
        
        if anno.H!=self.H or anno.W!=self.W:
            raise ValueError(f'Dimension mismatch {anno.H=}, {self.H=}, {anno.W=}, {self.W=}')
        if anno.bbox.y2>self.H or anno.bbox.x2>self.W:
            raise ValueError(f'Dimension mismatch {anno.bbox.y2=}, {self.H=}, {anno.bbox.x2=}, {self.W=}')

    def __post_init__(self):
        hutil.check_type(self.image, HImage)
        hutil.check_type(self.annos, HDataList)
        for i in range(len(self.annos)):
            self._check(i)

    def add_mask(self, x: HMaskAnno) -> None:
        hutil.check_type(x, HMaskAnno)
        self.annos.append(x)
        self._check(len(self.annos)-1)
