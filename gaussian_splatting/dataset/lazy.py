from collections import OrderedDict
from typing import Generic, List, Protocol, TypeVar

from gaussian_splatting.camera import Camera
from .dataset import CameraDataset


class CameraSpec(Protocol):
    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        ...


CameraSpecType = TypeVar("CameraSpecType", bound=CameraSpec)


class LazyCameraDataset(CameraDataset, Generic[CameraSpecType]):
    def __init__(self, camera_specs: List[CameraSpecType], load_mask: bool = True, load_depth: bool = True, device=None, cache_size: int = 2 ** 32):
        self.camera_specs = camera_specs
        self.load_mask = load_mask
        self.load_depth = load_depth
        self.device = device
        self.cache_size = cache_size
        self._cache: OrderedDict[int, Camera] = OrderedDict()
        self._cache_state = (self.load_mask, self.load_depth, self.device)

    def __len__(self) -> int:
        return len(self.camera_specs)

    def __getitem__(self, idx: int) -> Camera:
        state = (self.load_mask, self.load_depth, self.device)
        if state != self._cache_state:
            self._cache.clear()
            self._cache_state = state
        if self.cache_size <= 0:
            self._cache.clear()
            return self.camera_specs[idx].load(load_mask=self.load_mask, load_depth=self.load_depth, device=self.device)
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        camera = self.camera_specs[idx].load(load_mask=self.load_mask, load_depth=self.load_depth, device=self.device)
        self._cache[idx] = camera
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return camera

    def to(self, device) -> 'LazyCameraDataset':
        self.device = device
        self._cache.clear()
        self._cache_state = (self.load_mask, self.load_depth, self.device)
        return self
