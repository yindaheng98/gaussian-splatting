import os
from typing import NamedTuple, Callable
import torch
from PIL import Image
from .utils import fov2focal, focal2fov, getProjectionMatrix, getWorld2View2, read_image, matrix_to_quaternion


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
    custom_data: dict = {}


def camera2dict(camera: Camera, id):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = camera.R
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    pos = C2W[:3, 3]
    rot = C2W[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'width': camera.image_width,
        'height': camera.image_height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fx': fov2focal(camera.FoVx, camera.image_width),
        'fy': fov2focal(camera.FoVy, camera.image_height),
        'ground_truth_image_path': camera.ground_truth_image_path.replace("\\", "/"),
        "img_name": os.path.basename(camera.ground_truth_image_path),
    }
    return camera_entry


def build_camera(
        image_height: int, image_width: int,
        FoVx: float, FoVy: float,
        R: torch.Tensor, T: torch.Tensor,
        image_path: str = None, device="cuda"
):
    zfar = 100.0
    znear = 0.01
    trans = torch.zeros(3)
    scale = 1.0
    world_view_transform = getWorld2View2(R, T, trans, scale).to(device).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).to(device).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    quaternion = matrix_to_quaternion(R)
    gt_image = None
    if image_path is not None:
        gt_image = read_image(image_path).to(device)
        image_height = gt_image.shape[1]
        image_width = gt_image.shape[2]
    return Camera(
        # image_height=colmap_camera.image_height, # colmap_camera.image_height is read from cameras.bin, maybe dfferent from the actual image size
        # image_width=colmap_camera.image_width, # colmap_camera.image_width is read from cameras.bin, maybe dfferent from the actual image size
        image_height=image_height, image_width=image_width,
        FoVx=FoVx, FoVy=FoVy,
        R=R.to(device), T=T.to(device),
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        quaternion=quaternion.to(device),
        ground_truth_image_path=image_path,
        ground_truth_image=gt_image
    )


def dict2camera(camera_dict, device="cuda"):
    C2W = torch.zeros((4, 4))
    C2W[:3, 3] = torch.tensor(camera_dict['position'])
    C2W[:3, :3] = torch.tensor(camera_dict['rotation'])
    C2W[3, 3] = 1.0
    Rt = torch.linalg.inv(C2W)
    T = Rt[:3, 3]
    R = Rt[:3, :3]
    return build_camera(
        image_width=camera_dict['width'],
        image_height=camera_dict['height'],
        FoVx=focal2fov(camera_dict['fx'], camera_dict['width']),
        FoVy=focal2fov(camera_dict['fy'], camera_dict['height']),
        R=R,
        T=T,
        image_path=camera_dict['ground_truth_image_path'],
        device=device
    )
