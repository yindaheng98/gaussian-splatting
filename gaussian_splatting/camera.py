import os
from typing import NamedTuple, Callable
import torch
import logging
from .utils import fov2focal, focal2fov, getProjectionMatrix, getWorld2View2, read_image, read_depth, read_depth_mask, matrix_to_quaternion


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
    ground_truth_depth_path: str
    ground_truth_depth_mask_path: str
    ground_truth_image: torch.Tensor = None
    ground_truth_depth: torch.Tensor = None
    ground_truth_depth_mask: torch.Tensor = None
    postprocess: Callable[['Camera', torch.Tensor], torch.Tensor] = lambda camera, x: x
    bg_color: torch.Tensor = torch.tensor([0., 0., 0.])
    custom_data: dict = {}


def camera2dict(camera: Camera, id):
    Rt = torch.zeros((4, 4), device=camera.R.device)
    Rt[:3, :3] = camera.R
    Rt[:3, 3] = camera.T.to(camera.R.device)
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
        'ground_truth_image_path': camera.ground_truth_image_path.replace("\\", "/") if camera.ground_truth_image_path else None,
        'ground_truth_depth_path': camera.ground_truth_depth_path.replace("\\", "/") if camera.ground_truth_depth_path else None,
        'ground_truth_depth_mask_path': camera.ground_truth_depth_mask_path.replace("\\", "/") if camera.ground_truth_depth_mask_path else None,
        "img_name": os.path.basename(camera.ground_truth_image_path) if camera.ground_truth_image_path else None,
    }
    return camera_entry


def build_camera(
        image_height: int, image_width: int,
        FoVx: float, FoVy: float,
        R: torch.Tensor, T: torch.Tensor,
        image_path: str = None, depth_path: str = None, depth_mask_path: str = None,
        device="cuda", custom_data: dict = {}
):
    R, T = R.to(device=device, dtype=torch.float), T.to(device=device, dtype=torch.float)
    zfar = 100.0
    znear = 0.01
    trans = torch.zeros(3, device=R.device)
    scale = 1.0
    world_view_transform = getWorld2View2(R, T, trans, scale).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).to(device).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    quaternion = matrix_to_quaternion(R)
    gt_image = None
    if image_path is not None:
        gt_image = read_image(image_path).to(device)
        if gt_image.shape[1:] != (image_height, image_width):
            logging.warning(f"gt_image shape {gt_image.shape} does not match expected shape {image_height}x{image_width}, resizing.")
            gt_image = torch.nn.functional.interpolate(gt_image.unsqueeze(0), size=(image_height, image_width), mode='bilinear', align_corners=False).squeeze(0)
    gt_depth = None
    if depth_path is not None:
        if os.path.exists(depth_path):
            gt_depth = read_depth(depth_path).to(device)
            if gt_depth.shape != (image_height, image_width):
                logging.warning(f"gt_depth shape {gt_depth.shape} does not match expected shape {image_height}x{image_width}, resizing.")
                gt_depth = torch.nn.functional.interpolate(gt_depth.unsqueeze(0).unsqueeze(0), size=(image_height, image_width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        elif not os.path.exists(depth_path):
            logging.warning(f"Depth path {depth_path} does not exist, skipping depth loading.")
    gt_depth_mask = None
    if depth_mask_path is not None:
        if os.path.exists(depth_mask_path):
            gt_depth_mask = read_depth_mask(depth_mask_path).to(device)
            if gt_depth_mask.shape != (image_height, image_width):
                logging.warning(f"gt_depth_mask shape {gt_depth_mask.shape} does not match expected shape {image_height}x{image_width}, resizing.")
                gt_depth_mask = torch.nn.functional.interpolate(gt_depth_mask.unsqueeze(0).unsqueeze(0), size=(image_height, image_width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        elif not os.path.exists(depth_mask_path):
            logging.warning(f"Depth mask path {depth_mask_path} does not exist, skipping depth loading.")
    return Camera(
        # image_height=colmap_camera.image_height, # colmap_camera.image_height is read from cameras.bin, maybe dfferent from the actual image size
        # image_width=colmap_camera.image_width, # colmap_camera.image_width is read from cameras.bin, maybe dfferent from the actual image size
        image_height=image_height, image_width=image_width,
        FoVx=FoVx, FoVy=FoVy,
        R=R, T=T,
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        quaternion=quaternion,
        ground_truth_image_path=image_path,
        ground_truth_image=gt_image,
        ground_truth_depth_path=depth_path,
        ground_truth_depth=gt_depth,
        ground_truth_depth_mask_path=depth_mask_path,
        ground_truth_depth_mask=gt_depth_mask,
        custom_data=custom_data,
    )


def dict2camera(camera_dict, load_depth=False, device="cuda", custom_data: dict = {}):
    C2W = torch.zeros((4, 4), device=device)
    C2W[:3, 3] = torch.tensor(camera_dict['position'], dtype=torch.float, device=device)
    C2W[:3, :3] = torch.tensor(camera_dict['rotation'], dtype=torch.float, device=device)
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
        image_path=camera_dict['ground_truth_image_path'] if 'ground_truth_image_path' in camera_dict else None,
        depth_path=camera_dict['ground_truth_depth_path'] if (load_depth and 'ground_truth_depth_path' in camera_dict) else None,
        depth_mask_path=camera_dict['ground_truth_depth_mask_path'] if (load_depth and 'ground_truth_depth_mask_path' in camera_dict) else None,
        device=device,
        custom_data=custom_data,
    )
