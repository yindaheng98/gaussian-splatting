import json
from typing import List
from gaussian_splatting import Camera
import torch
import torch.nn as nn

from gaussian_splatting.camera import camera2dict
from gaussian_splatting.utils import normalize_quaternion, quaternion_to_matrix
from .dataset import CameraDataset, JSONCameraDataset


def exposure_postprocess(camera: Camera, x: torch.Tensor):
    exposure = camera.custom_data['exposures']
    return torch.matmul(x.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]


class TrainableCameraDataset(CameraDataset):

    def __init__(self, cameras: List[Camera], exposures: List[torch.Tensor] = []):
        super().__init__()
        self.cameras = cameras
        self.quaternions = nn.Parameter(torch.stack([camera.quaternion for camera in cameras]))
        self.Ts = nn.Parameter(torch.stack([camera.T for camera in cameras]))
        self.exposures = nn.Parameter(torch.stack([torch.eye(3, 4, device=camera.T.device) for camera in cameras]))
        if len(exposures) > 0:
            assert len(exposures) == len(cameras), "Number of exposures must match number of cameras"
            with torch.no_grad():
                for idx, exposure in enumerate(exposures):
                    self.exposures[idx, ...] = exposure.to(self.exposures.device)

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Camera:
        viewpoint_camera: Camera = self.cameras[idx]
        rel_w2c = torch.eye(4, device=self.quaternions.device)
        quaternion = self.quaternions[idx, ...]
        rel_w2c[:3, :3] = quaternion_to_matrix(normalize_quaternion(quaternion.unsqueeze(0))).squeeze(0)
        rel_w2c[:3, 3] = self.Ts[idx, ...]

        # use the computed camera matrices rather than the ones from the dataset
        w2c = rel_w2c.T
        projmatrix = (
            w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        campos = w2c.inverse()[3, :3]

        return Camera(**{
            **self.cameras[idx]._asdict(),
            'world_view_transform': w2c,
            'full_proj_transform': projmatrix,
            'camera_center': campos,
            'quaternion': self.quaternions[idx, ...],
            'T': self.Ts[idx, ...],
            'postprocess': exposure_postprocess,
            'custom_data': {
                **self.cameras[idx].custom_data,
                'exposures': self.exposures[idx, ...]
            }
        })

    def to(self, device):
        self.quaternions.to(device)
        self.Ts.to(device)
        self.exposures.to(device)
        return self

    def save_cameras(self, path):
        cameras = []
        for idx, camera in enumerate(self):
            cameras.append({
                **camera2dict(Camera(**{
                    **camera._asdict(),
                    'R': quaternion_to_matrix(self.quaternions[idx, ...]),
                    'T': self.Ts[idx, ...],
                }), idx),
                "exposure": self.exposures[idx, ...].detach().tolist(),
            })
        with open(path, 'w') as f:
            json.dump(cameras, f, indent=2)

    @classmethod
    def from_json(cls, path, load_depth=False):
        cameras = JSONCameraDataset(path, load_depth=load_depth)
        exposures = [(torch.tensor(camera['exposure'], dtype=torch.float) if 'exposure' in camera else torch.eye(3, 4)) for camera in cameras.json_cameras]
        return cls(cameras, exposures)
