import numpy as np
import torch
import os
from tqdm import tqdm
import open3d as o3d
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.render import prepare_rendering
from gaussian_splatting.utils import fov2focal


def depth2xyz(depth: torch.Tensor, K: torch.Tensor, R_c2w: torch.Tensor, T_c2w: torch.Tensor) -> torch.Tensor:
    height, width = depth.shape
    uv = torch.ones((3, height, width), dtype=torch.float32, device=depth.device)
    uv[0, ...] = torch.arange(0, width, dtype=torch.float32).unsqueeze(0).expand(height, -1)
    uv[1, ...] = torch.arange(0, height, dtype=torch.float32).unsqueeze(1).expand(-1, width)
    xyz_camera = torch.inverse(K) @ uv.reshape(3, -1) * depth.reshape(-1)
    xyz_world = R_c2w @ xyz_camera + T_c2w.unsqueeze(1)
    return xyz_world.T.reshape(*uv.shape)


def build_K(FoVx, FoVy, width, height):
    fx = fov2focal(FoVx, width)
    fy = fov2focal(FoVy, height)
    K = torch.tensor([
        [fx, 0, width/2],
        [0, fy, height/2],
        [0, 0, 1]
    ])
    return K


def build_pcd(color: torch.Tensor, depth: torch.Tensor, FoVx, FoVy, width, height, R_c2w: torch.Tensor, T_c2w: torch.Tensor) -> torch.Tensor:
    assert color.shape[-2:] == depth.shape[-2:], ValueError("Size of depth map should match color image")
    K = build_K(FoVx, FoVy, width, height).to(depth.device)
    xyz = depth2xyz(depth, K, R_c2w, T_c2w).permute(1, 2, 0)
    color = color.permute(1, 2, 0) * 255
    pcd = o3d.geometry.PointCloud()
    idx = torch.abs(xyz).sum(axis=-1) < 1000
    pcd.points = o3d.utility.Vector3dVector(xyz[idx, ...].cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color[idx, ...].cpu().numpy().astype(np.float32)/255)
    return pcd


def rendering(dataset: CameraDataset, gaussians: GaussianModel, save: str):
    pbar = tqdm(dataset, desc="Rendering progress")
    for idx, camera in enumerate(pbar):
        out = gaussians(camera)
        rendering = out["render"]
        gt = camera.ground_truth_image
        depth = out["depth"]
        depth_gt = camera.ground_truth_depth
        pcd = build_pcd(rendering, 1 / depth.squeeze(0), camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.R, camera.T)
        pcd_gt = build_pcd(gt, depth_gt, camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.R, camera.T)
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "densify", "camera", "camera-densify"], default="pure")
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device, mode=args.mode,
            load_ply=load_ply, load_camera=args.load_camera, with_depth=True)
        rendering(dataset, gaussians, save)
