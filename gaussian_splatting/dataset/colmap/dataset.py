import os
from typing import List, NamedTuple
import logging
import numpy as np
from PIL import Image

import torch

from gaussian_splatting.camera import Camera, build_camera
from ..lazy import LazyCameraDataset
from ..camera_trainable import TrainableCameraDataset
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
    cx: float
    cy: float
    R: torch.Tensor
    T: torch.Tensor
    image_path: str
    image_mask_path: str
    depth_path: str
    depth_mask_path: str

    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        return build_camera(
            image_height=self.image_height,
            image_width=self.image_width,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            cx=self.cx,
            cy=self.cy,
            R=self.R,
            T=self.T,
            image_path=self.image_path,
            image_mask_path=self.image_mask_path if load_mask else None,
            depth_path=self.depth_path if load_depth else None,
            depth_mask_path=self.depth_mask_path if load_depth else None,
            device=device,
        )


def parse_colmap_camera(cameras, images, image_dir, load_mask=True, image_mask_dir=None, depth_dir=None, depth_mask_dir=None) -> List[ColmapCamera]:
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
            cx = float(intr.params[1])
            cy = float(intr.params[2])
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = float(intr.params[2])
            cy = float(intr.params[3])
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            raise ValueError("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")

        image_path = os.path.join(image_dir, extr.name)
        with Image.open(image_path) as img:
            width_file, height_file = img.size
            if width_file != width or height_file != height:
                logging.warning(f"Image size mismatch for {image_path}: expected ({width}, {height}), got ({width_file}, {height_file}) in file. Using size from file.")
                # FoV is converted back to focal length with the file size, which
                # scales fx and fy by the same ratio. Scale the principal point too.
                cx *= width_file / float(width)
                cy *= height_file / float(height)
                width, height = width_file, height_file
        image_mask_path = None
        if load_mask:
            image_mask_path = os.path.join(image_mask_dir, extr.name + '.tiff')
            if not os.path.exists(image_mask_path):
                image_mask_path = os.path.splitext(image_mask_path)[0] + '.png'
        depth_path, depth_mask_path = None, None
        if depth_dir is not None:
            depth_path = os.path.join(depth_dir, extr.name + '.tiff')
            if not os.path.exists(depth_path):
                depth_path = os.path.splitext(depth_path)[0] + '.png'
            depth_mask_path = os.path.join(depth_mask_dir, extr.name + '.tiff')
            if not os.path.exists(depth_mask_path):
                depth_mask_path = os.path.splitext(depth_mask_path)[0] + '.png'
        parsed_cameras.append(ColmapCamera(
            image_height=height, image_width=width,
            R=torch.from_numpy(R).type(torch.float), T=torch.from_numpy(T).type(torch.float),
            FoVy=FovY, FoVx=FovX, cx=cx, cy=cy,
            image_path=image_path,
            image_mask_path=image_mask_path,
            depth_path=depth_path,
            depth_mask_path=depth_mask_path,
        ))
    return parsed_cameras


def read_colmap_cameras(colmap_folder, load_mask=True, load_depth=True) -> List[ColmapCamera]:
    path = colmap_folder
    image_dir = os.path.join(path, "images")
    image_mask_dir = os.path.join(path, "image_masks") if load_mask else None
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
    depth_mask_dir = os.path.join(path, "depth_masks") if load_depth else None
    return parse_colmap_camera(
        cam_extrinsics, cam_intrinsics, image_dir,
        load_mask=load_mask, image_mask_dir=image_mask_dir,
        depth_dir=depth_dir, depth_mask_dir=depth_mask_dir,
    )


class ColmapCameraDataset(LazyCameraDataset[ColmapCamera]):
    def __init__(self, colmap_folder, load_mask=True, load_depth=True, cache_size: int = 2 ** 32):
        super().__init__(
            read_colmap_cameras(colmap_folder, load_mask=load_mask, load_depth=load_depth),
            load_mask=load_mask,
            load_depth=load_depth,
            cache_size=cache_size,
        )


def ColmapTrainableCameraDataset(colmap_folder, load_mask=True, load_depth=True, cache_size: int = 2 ** 32):
    return TrainableCameraDataset(ColmapCameraDataset(colmap_folder, load_mask=load_mask, load_depth=load_depth, cache_size=cache_size))
