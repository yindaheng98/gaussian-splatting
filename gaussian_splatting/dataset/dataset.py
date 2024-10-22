from abc import abstractmethod
import json

from gaussian_splatting.camera import Camera, camera2dict
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
