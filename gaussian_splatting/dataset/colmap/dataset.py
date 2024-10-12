import os
import numpy as np
from typing import NamedTuple

from gaussian_splatting.utils import focal2fov
from .loader import (
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


class ColmapCameras:
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
