import os
import numpy as np

import torch
from PIL import Image

from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset, RawCamera
from gaussian_splatting.utils import focal2fov, getProjectionMatrix, getWorld2View2, PILtoTorch
from .utils import (
    read_extrinsics_text, read_extrinsics_binary,
    read_intrinsics_text, read_intrinsics_binary,
    qvec2rotmat
)


def read_colmap_cameras(colmap_folder):
    cameras = []
    path = colmap_folder
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    for _, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            raise ValueError("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")

        image_path = os.path.join(os.path.join(path, "images"), extr.name)
        cameras.append(RawCamera(
            image_height=height, image_width=width,
            R=torch.from_numpy(R), T=torch.from_numpy(T),
            FoVy=FovY, FoVx=FovX,
            image_path=image_path
        ))
    return cameras


def parse_RawCamera(colmap_camera: RawCamera, device="cuda"):
    zfar = 100.0
    znear = 0.01
    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    R = colmap_camera.R
    T = colmap_camera.T
    FoVx = colmap_camera.FoVx
    FoVy = colmap_camera.FoVy
    world_view_transform = torch.tensor(getWorld2View2(R.numpy(), T.numpy(), trans, scale), device=device).transpose(0, 1)
    projection_matrix = torch.tensor(getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy), device=device).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    pil_image = Image.open(colmap_camera.image_path)
    torch_image = PILtoTorch(pil_image)
    gt_image = torch_image[:3, ...].clamp(0.0, 1.0).to(device)
    image_height = gt_image.shape[1]
    image_width = gt_image.shape[2]
    return Camera(
        # image_height=colmap_camera.image_height, # colmap_camera.image_height is read from cameras.bin, maybe dfferent from the actual image size
        # image_width=colmap_camera.image_width, # colmap_camera.image_width is read from cameras.bin, maybe dfferent from the actual image size
        image_height=image_height, image_width=image_width,
        FoVx=FoVx, FoVy=FoVy,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
        ground_truth_image=gt_image
    )


class ColmapCameraDataset(CameraDataset):
    def __init__(self, colmap_folder, device="cuda"):
        super().__init__()
        self.raw_cameras = read_colmap_cameras(colmap_folder)
        self.cameras = [parse_RawCamera(cam, device=device) for cam in self.raw_cameras]

    def to(self, device):
        self.cameras = [parse_RawCamera(cam, device=device) for cam in self.raw_cameras]
        return self

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]
