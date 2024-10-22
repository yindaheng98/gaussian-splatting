from abc import abstractmethod
import json

from gaussian_splatting.camera import Camera, camera2dict, dict2camera
from torch.utils.data import Dataset


class CameraDataset(Dataset):

    @abstractmethod
    def to(self, device) -> 'CameraDataset':
        return self

    @abstractmethod
    def __getitem__(self, idx) -> Camera:
        pass

    def save_cameras(self, path):
        cameras = []
        for id, camera in enumerate(self):
            cameras.append(camera2dict(camera, id))
        with open(path, 'w') as f:
            json.dump(cameras, f, indent=2)


class JSONCameraDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.json_cameras = json.load(f)
        self.cameras = [dict2camera(camera) for camera in self.json_cameras]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

    def to(self, device):
        self.cameras = [dict2camera(camera, device=device) for camera in self.json_cameras]
        return self
