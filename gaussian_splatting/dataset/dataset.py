from abc import abstractmethod
from typing import List
from gaussian_splatting import Camera
from torch.utils.data import Dataset


class CameraDataset(Dataset):

    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    @abstractmethod
    def to(self, device) -> 'CameraDataset':
        pass

    def __len__(self):
        return len(self.cameras)
