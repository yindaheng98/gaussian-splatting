import os
from typing import List, NamedTuple
import numpy as np

import torch

from gaussian_splatting.camera import build_camera
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from gaussian_splatting.utils import focal2fov
from .read_write_model import (
    read_cameras_text, read_cameras_binary,
    read_images_text, read_images_binary,
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
    depth_path: str
    depth_mask_path: str


def parse_colmap_camera(cameras, images, image_dir, depth_dir=None) -> List[ColmapCamera]:
    parsed_cameras = []
    for _, key in enumerate(cameras):
        extr = cameras[key]
        intr = images[extr.camera_id]
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
        depth_path, depth_mask_path = None, None
        if depth_dir is not None:
            depth_path = os.path.join(depth_dir, os.path.splitext(extr.name)[0] + '.tiff')
            if not os.path.exists(depth_path):
                depth_path = os.path.splitext(depth_path)[0] + '.png'
            depth_mask_path = os.path.join(depth_dir, os.path.splitext(extr.name)[0] + '_mask.tiff')
            if not os.path.exists(depth_mask_path):
                depth_mask_path = os.path.splitext(depth_mask_path)[0] + '.png'
        parsed_cameras.append(ColmapCamera(
            image_height=height, image_width=width,
            R=torch.from_numpy(R).type(torch.float), T=torch.from_numpy(T).type(torch.float),
            FoVy=FovY, FoVx=FovX,
            image_path=image_path,
            depth_path=depth_path,
            depth_mask_path=depth_mask_path,
        ))
    return parsed_cameras


def read_colmap_cameras(colmap_folder, load_depth=False) -> List[ColmapCamera]:
    path = colmap_folder
    image_dir = os.path.join(path, "images")
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_images_binary(cameras_extrinsic_file)
        cam_intrinsics = read_cameras_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_images_text(cameras_extrinsic_file)
        cam_intrinsics = read_cameras_text(cameras_intrinsic_file)
    depth_dir = os.path.join(path, "depths") if load_depth else None
    return parse_colmap_camera(cam_extrinsics, cam_intrinsics, image_dir, depth_dir)


class ColmapCameraDataset(CameraDataset):
    def __init__(self, colmap_folder, load_depth=False):
        super().__init__()
        self.raw_cameras = read_colmap_cameras(colmap_folder, load_depth=load_depth)
        self.cameras = [build_camera(**cam._asdict()) for cam in self.raw_cameras]

    def to(self, device):
        self.cameras = [build_camera(**cam._asdict(), device=device) for cam in self.raw_cameras]
        return self

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]


def ColmapTrainableCameraDataset(colmap_folder, load_depth=False):
    return TrainableCameraDataset(ColmapCameraDataset(colmap_folder, load_depth=load_depth))
