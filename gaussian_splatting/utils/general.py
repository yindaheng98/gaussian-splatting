import torch
import numpy as np
from PIL import Image
import cv2
import tifffile


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def PILtoTorch(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)).type(torch.float) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def read_image(image_path):
    pil_image = Image.open(image_path)
    torch_image = PILtoTorch(pil_image)
    return torch_image[:3, ...].clamp(0.0, 1.0)


def read_png_depth(depth_path):
    cv2_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if cv2_depth.ndim != 2:
        cv2_depth = cv2_depth[..., 0]
    torch_image = torch.from_numpy(cv2_depth).type(torch.float)
    return torch_image


def read_tiff_depth(depth_path):
    tiff_depth = tifffile.imread(depth_path)
    if tiff_depth.ndim != 2:
        tiff_depth = tiff_depth[..., 0]
    torch_image = torch.from_numpy(tiff_depth).type(torch.float)
    return torch_image


def read_depth(depth_path):
    if depth_path.endswith('.tiff'):
        return read_tiff_depth(depth_path)
    return read_png_depth(depth_path)


def read_depth_mask(depth_path):
    if depth_path.endswith('.tiff'):
        return read_tiff_depth(depth_path)
    return read_png_depth(depth_path) / 255.0


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0]*r[:, 0] + r[:, 1]*r[:, 1] + r[:, 2]*r[:, 2] + r[:, 3]*r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), device=s.device)
    R = build_rotation(r.to(s.device))

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L
