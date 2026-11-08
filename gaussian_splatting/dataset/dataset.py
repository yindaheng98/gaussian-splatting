from abc import abstractmethod
from collections import OrderedDict
import json
from typing import Generic, List, Protocol, TypeVar
import torch

from gaussian_splatting.utils import getWorld2View2
from gaussian_splatting.camera import Camera, camera2dict, dict2camera


class CameraDataset:

    @abstractmethod
    def to(self, device) -> 'CameraDataset':
        return self

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx) -> Camera:
        raise NotImplementedError

    def save_cameras(self, path):
        cameras = []
        for id, camera in enumerate(self):
            cameras.append(camera2dict(camera, id))
        with open(path, 'w') as f:
            json.dump(cameras, f, indent=2)

    def scene_extent(dataset):
        nerf_normalization = getNerfppNorm(dataset)
        scene_extent = nerf_normalization["radius"]
        return scene_extent.item()


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


class JSONCameraSpec(dict):
    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        return dict2camera(self, load_mask=load_mask, load_depth=load_depth, device=device)


class JSONCameraDataset(LazyCameraDataset[JSONCameraSpec]):
    def __init__(self, path, load_mask=True, load_depth=True, cache_size: int = 2 ** 32):
        with open(path, 'r') as f:
            json_cameras = json.load(f)
        super().__init__([JSONCameraSpec(camera) for camera in json_cameras], load_mask=load_mask, load_depth=load_depth, cache_size=cache_size)


def getNerfppNorm(cameras: List[Camera]):
    def get_center_and_diag(cam_centers):
        cam_centers = torch.hstack(cam_centers)
        avg_cam_center = torch.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = torch.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = torch.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cameras:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = torch.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
