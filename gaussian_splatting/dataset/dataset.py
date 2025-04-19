from abc import abstractmethod
import json
from typing import List
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


class JSONCameraDataset(CameraDataset):
    def __init__(self, path, load_depth=False):
        with open(path, 'r') as f:
            self.json_cameras = json.load(f)
        self.load_depth = load_depth
        self.cameras = [dict2camera(camera, load_depth=self.load_depth) for camera in self.json_cameras]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

    def to(self, device):
        self.cameras = [dict2camera(camera, load_depth=self.load_depth, device=device) for camera in self.json_cameras]
        return self


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
