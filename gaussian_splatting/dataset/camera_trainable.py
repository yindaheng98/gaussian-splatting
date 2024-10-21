from typing import List
from gaussian_splatting import Camera
import torch
import torch.nn as nn
from .dataset import CameraDataset


class TrainableCameraDataset(CameraDataset):

    def __init__(self, cameras: List[Camera]):
        super().__init__()
        self.cameras = cameras
        self.quaternions = nn.Parameter(torch.stack([camera.quaternion for camera in cameras]))
        self.Ts = nn.Parameter(torch.stack([camera.T for camera in cameras]))

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Camera:
        return Camera(**{**self.cameras[idx]._asdict(), 'R': self.Rs[idx, ...], 'T': self.Ts[idx, ...]})

    def to(self, device):
        self.quaternions.to(device)
        self.Ts.to(device)
        return self
