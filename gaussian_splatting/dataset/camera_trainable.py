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
        self.exposures = nn.Parameter(torch.stack([torch.eye(3, 4, device=camera.T.device) for camera in cameras]))

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Camera:

        def postprocess(_, x):
            return torch.matmul(x.permute(1, 2, 0), self.exposures[idx, :3, :3]).permute(2, 0, 1) + self.exposures[idx, :3, 3, None, None]
        return Camera(**{
            **self.cameras[idx]._asdict(),
            'quaternion': self.quaternions[idx, ...],
            'T': self.Ts[idx, ...],
            'postprocess': postprocess
        })

    def to(self, device):
        self.quaternions.to(device)
        self.Ts.to(device)
        self.exposures.to(device)
        return self
