import json

import torch

from gaussian_splatting.camera import Camera, camera2dict, dict2camera
from .exposure import exposure_postprocess
from .lazy import LazyCameraDataset


class JSONCameraSpec(dict):
    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        return dict2camera(self, load_mask=load_mask, load_depth=load_depth, device=device)


class JSONCameraDataset(LazyCameraDataset[JSONCameraSpec]):
    def __init__(self, path, load_mask=True, load_depth=True, cache_size: int = 2 ** 32):
        with open(path, 'r') as f:
            json_cameras = json.load(f)
        super().__init__([JSONCameraSpec(camera) for camera in json_cameras], load_mask=load_mask, load_depth=load_depth, cache_size=cache_size)

    def __getitem__(self, idx) -> Camera:
        camera = super().__getitem__(idx)
        json_camera = self.camera_specs[idx]
        device = camera.K.device
        return camera._replace(
            postprocess=exposure_postprocess,
            custom_data={
                **camera.custom_data,
                'exposures': (torch.tensor(json_camera['exposure'], dtype=torch.float, device=device) if 'exposure' in json_camera else torch.eye(3, 4, dtype=torch.float, device=device))
            }
        )

    def save_cameras(self, path):
        cameras = []
        for idx, camera in enumerate(self):
            cameras.append({
                **camera2dict(camera, idx),
                "exposure": camera.custom_data['exposures'].detach().tolist(),
            })
        with open(path, 'w') as f:
            json.dump(cameras, f, indent=2)
