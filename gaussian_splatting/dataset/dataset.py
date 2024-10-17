from abc import abstractmethod
from typing import List, NamedTuple
from gaussian_splatting import Camera
import torch
import torch.nn as nn
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

    @abstractmethod
    def to(self, device) -> 'CameraDataset':
        return self


class TrainableCameraDataset(CameraDataset):

    def __init__(self, cameras: List[RawCamera]):
        super().__init__()
        self.cameras = cameras
        self.Rs = nn.Parameter(torch.stack([camera.R for camera in cameras]))
        self.Ts = nn.Parameter(torch.stack([camera.T for camera in cameras]))

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> RawCamera:
        return RawCamera(**self.cameras[idx], R=self.Rs[idx, ...], T=self.Ts[idx, ...])

    def to(self, device):
        self.Rs.to(device)
        self.Ts.to(device)
        return self
