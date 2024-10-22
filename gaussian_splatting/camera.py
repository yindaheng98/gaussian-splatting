import os
from typing import NamedTuple, Callable
import torch
from .utils import fov2focal


class Camera(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: torch.Tensor
    T: torch.Tensor
    world_view_transform: torch.Tensor
    projection_matrix: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    quaternion: torch.Tensor
    ground_truth_image_path: str
    ground_truth_image: torch.Tensor = None
    postprocess: Callable[['Camera', torch.Tensor], torch.Tensor] = lambda camera, x: x
    bg_color: torch.Tensor = torch.tensor([0., 0., 0.])


def camera2dict(camera: Camera, id):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose(0, 1)
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = torch.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'width': camera.image_width,
        'height': camera.image_height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FoVx, camera.image_height),
        'fx': fov2focal(camera.FoVy, camera.image_width),
        'ground_truth_image_path': camera.ground_truth_image_path.replace("\\", "/"),
        "img_name": os.path.basename(camera.ground_truth_image_path),
    }
    return camera_entry
