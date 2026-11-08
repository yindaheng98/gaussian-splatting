import json

from gaussian_splatting.camera import Camera, dict2camera
from .lazy import LazyCameraDataset


class JSONCameraSpec(dict):
    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        return dict2camera(self, load_mask=load_mask, load_depth=load_depth, device=device)


class JSONCameraDataset(LazyCameraDataset[JSONCameraSpec]):
    def __init__(self, path, load_mask=True, load_depth=True, cache_size: int = 2 ** 32):
        with open(path, 'r') as f:
            json_cameras = json.load(f)
        super().__init__([JSONCameraSpec(camera) for camera in json_cameras], load_mask=load_mask, load_depth=load_depth, cache_size=cache_size)
