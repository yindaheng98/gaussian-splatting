import os
import numpy as np
from typing import NamedTuple

import torch
from PIL import Image

from gaussian_splatting import Camera
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.utils import focal2fov, getProjectionMatrix, getWorld2View2, PILtoTorch
from .utils import (
    read_extrinsics_text, read_extrinsics_binary,
    read_intrinsics_text, read_intrinsics_binary,
    qvec2rotmat
)


class ColmapCamera(NamedTuple):
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: np.array
    T: np.array
    image_path: str


class ColmapCameraReader:
    def __init__(self, colmap_folder):
        self.colmap_folder = colmap_folder
        self.cameras = self._read_cameras()

    def _read_cameras(self):
        cameras = []
        path = self.colmap_folder
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
            cameras.append(ColmapCamera(
                image_height=height,
                image_width=width,
                R=R, T=T, FoVx=FovX,
                FoVy=FovY,
                image_path=image_path
            ))
        return cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]


def ColmapCamera2DatasetCamera(colmap_camera: ColmapCamera, device="cuda"):
    zfar = 100.0
    znear = 0.01
    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    R = colmap_camera.R
    T = colmap_camera.T
    FoVx = colmap_camera.FoVx
    FoVy = colmap_camera.FoVy
    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), device=device).transpose(0, 1)
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


class ColmapCameraDataset(ColmapCameraReader, CameraDataset):
    def __init__(self, colmap_folder, device="cuda"):
        super().__init__(colmap_folder)
        self.to(device)

    def to(self, device):
        self.device_cameras = [ColmapCamera2DatasetCamera(cam, device=device) for cam in self.cameras]
        return self

    def __getitem__(self, idx):
        return self.device_cameras[idx]
