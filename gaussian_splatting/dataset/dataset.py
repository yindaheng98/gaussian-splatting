from abc import abstractmethod
from typing import List, NamedTuple
from gaussian_splatting import Camera
import torch
from torch.utils.data import Dataset


class RawCamera(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor
    T: torch.Tensor
    image_path: str


class CameraDataset(Dataset):

    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    @abstractmethod
    def to(self, device) -> 'CameraDataset':
        pass

    def __len__(self):
        return len(self.cameras)
