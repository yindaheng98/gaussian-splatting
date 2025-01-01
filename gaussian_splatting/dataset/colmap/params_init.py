import os
import struct
import numpy as np
import torch
from plyfile import PlyData

from gaussian_splatting import GaussianModel


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def read_points3D_ply(path_to_model_file):
    plydata = PlyData.read(path_to_model_file)
    vertices = plydata['vertex']
    xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    return xyz, rgb, None


def read_colmap_points3D(colmap_folder: str):
    init_path = os.path.join(colmap_folder, "sparse/0", "points3D.ply")
    if os.path.exists(init_path):
        xyz, rgb, errors = read_points3D_ply(init_path)
        return xyz, rgb, errors
    try:
        init_path = os.path.join(colmap_folder, "sparse/0", "points3D.bin")
        xyz, rgb, errors = read_points3D_binary(init_path)
    except:
        init_path = os.path.join(colmap_folder, "sparse/0", "points3D.txt")
        xyz, rgb, errors = read_points3D_text(init_path)
    return xyz, rgb, errors


def colmap_init(model: GaussianModel, colmap_folder: str):
    with torch.no_grad():
        xyz, rgb, _ = read_colmap_points3D(colmap_folder)
        return model.create_from_pcd(torch.from_numpy(xyz), torch.from_numpy(rgb) / 255.0)
