import os
from typing import NamedTuple
import numpy as np

import torch

from gaussian_splatting import CameraTrainableGaussianModel
from gaussian_splatting.camera import build_camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.utils import focal2fov
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
    R: torch.Tensor
    T: torch.Tensor
    image_path: str


def parse_colmap_camera(cam_extrinsics, cam_intrinsics, image_dir):
    cameras = []
    for _, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        R = qvec2rotmat(extr.qvec)
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

        image_path = os.path.join(image_dir, extr.name)
        cameras.append(ColmapCamera(
            image_height=height, image_width=width,
            R=torch.from_numpy(R), T=torch.from_numpy(T),
            FoVy=FovY, FoVx=FovX,
            image_path=image_path
        ))
    return cameras


def read_cameras_binary(cameras_extrinsic_file, cameras_intrinsic_file, image_dir):
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    return parse_colmap_camera(cam_extrinsics, cam_intrinsics, image_dir)


def read_cameras_text(cameras_extrinsic_file, cameras_intrinsic_file, image_dir):
    cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    return parse_colmap_camera(cam_extrinsics, cam_intrinsics, image_dir)


def read_colmap_cameras(colmap_folder):
    path = colmap_folder
    image_dir = os.path.join(path, "images")
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        return read_cameras_binary(cameras_extrinsic_file, cameras_intrinsic_file, image_dir)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        return read_cameras_text(cameras_extrinsic_file, cameras_intrinsic_file, image_dir)


class ColmapCameraDataset(CameraDataset):
    def __init__(self, colmap_folder):
        super().__init__()
        self.raw_cameras = read_colmap_cameras(colmap_folder)
        self.cameras = [build_camera(**cam._asdict()) for cam in self.raw_cameras]

    def to(self, device):
        self.cameras = [build_camera(**cam._asdict(), device=device) for cam in self.raw_cameras]
        return self

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]


def ColmapTrainableCameraDataset(colmap_folder):
    return TrainableCameraDataset(ColmapCameraDataset(colmap_folder))


def ColmapCameraTrainableGaussianModel(colmap_folder, *args, **kwargs):
    return CameraTrainableGaussianModel(dataset=ColmapTrainableCameraDataset(colmap_folder), *args, **kwargs)
