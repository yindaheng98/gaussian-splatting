import os
import numpy as np
import torch
from plyfile import PlyData

from gaussian_splatting import GaussianModel
from .read_write_model import read_points3D_text, read_points3D_binary


def read_points3D_ply(path_to_model_file):
    plydata = PlyData.read(path_to_model_file)
    vertices = plydata['vertex']
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    return xyz, rgb, None


def read_colmap_points3D(colmap_folder: str):
    init_path = os.path.join(colmap_folder, "sparse/0", "points3D.ply")
    if os.path.exists(init_path):
        xyz, rgb, error = read_points3D_ply(init_path)
        return xyz, rgb, error
    try:
        init_path = os.path.join(colmap_folder, "sparse/0", "points3D.bin")
        points3D = read_points3D_binary(init_path)
    except:
        init_path = os.path.join(colmap_folder, "sparse/0", "points3D.txt")
        points3D = read_points3D_text(init_path)
    pts_indices = np.array([points3D[key].id for key in points3D])
    xyz = np.zeros([pts_indices.max()+1, 3])
    rgb = np.zeros([pts_indices.max()+1, 3])
    error = np.zeros([pts_indices.max()+1])
    xyz[pts_indices] = np.array([points3D[key].xyz for key in points3D])
    rgb[pts_indices] = np.array([points3D[key].rgb for key in points3D])
    error[pts_indices] = np.array([points3D[key].error for key in points3D])
    return xyz[pts_indices], rgb[pts_indices], error[pts_indices]


def colmap_init(model: GaussianModel, colmap_folder: str):
    with torch.no_grad():
        xyz, rgb, _ = read_colmap_points3D(colmap_folder)
        return model.create_from_pcd(torch.from_numpy(xyz).type(torch.float), torch.from_numpy(rgb).type(torch.float) / 255.0)
