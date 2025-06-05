from typing import Tuple
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
import tifffile
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.utils import psnr, unproject
from gaussian_splatting.utils.lpipsPyTorch import lpips
from gaussian_splatting.prepare import prepare_dataset, prepare_gaussians


def prepare_rendering(sh_degree: int, source: str, device: str, trainable_camera: bool = False, load_ply: str = None, load_camera: str = None, load_depth=False) -> Tuple[CameraDataset, GaussianModel]:
    dataset = prepare_dataset(source=source, device=device, trainable_camera=trainable_camera, load_camera=load_camera, load_depth=load_depth)
    gaussians = prepare_gaussians(sh_degree=sh_degree, source=source, device=device, trainable_camera=trainable_camera, load_ply=load_ply)
    return dataset, gaussians


def build_pcd(color: torch.Tensor, invdepth: torch.Tensor, mask: torch.Tensor, FoVx, FoVy) -> torch.Tensor:
    assert color.shape[-2:] == invdepth.shape[-2:], ValueError("Size of depth map should match color image")
    xyz = unproject(1 / invdepth, FoVx, FoVy)
    color = color.permute(1, 2, 0)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[mask, ...].cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color[mask, ...].cpu().numpy())
    return pcd


def build_pcd_rescale(
        color: torch.Tensor, color_gt: torch.Tensor,
        invdepth: torch.Tensor, invdepth_gt: torch.Tensor, mask: torch.Tensor,
        FoVx, FoVy,
        rescale_depth_gt=True) -> torch.Tensor:
    invdepth_gt_rescale = invdepth_gt
    mask = (mask > 1e-6)
    if rescale_depth_gt:
        mean_gt, std_gt = invdepth_gt.mean(), invdepth_gt.std()
        mean, std = invdepth.mean(), invdepth.std()
        invdepth_gt_rescale = (invdepth_gt - mean_gt) / std_gt * std + mean
    pcd = build_pcd(color, invdepth, mask, FoVx, FoVy)
    pcd_gt = build_pcd(color_gt, invdepth_gt_rescale, mask, FoVx, FoVy)
    return pcd, pcd_gt, invdepth_gt_rescale


def rendering(
        dataset: CameraDataset, gaussians: GaussianModel, save: str, save_pcd: bool = False,
        rescale_depth_gt: bool = True) -> None:
    render_path = os.path.join(save, "renders")
    gt_path = os.path.join(save, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    pbar = tqdm(dataset, desc="Rendering progress")
    for idx, camera in enumerate(pbar):
        out = gaussians(camera)
        rendering = out["render"]
        gt = camera.ground_truth_image
        pbar.set_postfix({"PSNR": psnr(rendering, gt).mean().item(), "LPIPS": lpips(rendering, gt).mean().item()})
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))
        depth = out["depth"].squeeze(0)
        tifffile.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.tiff"), depth.cpu().numpy())
        if save_pcd:
            import open3d as o3d
            if camera.ground_truth_depth is not None:
                mask = camera.ground_truth_depth_mask if camera.ground_truth_depth_mask is not None else torch.ones_like(camera.ground_truth_depth)
                pcd, pcd_gt, invdepth_gt_rescale = build_pcd_rescale(rendering, gt, depth, camera.ground_truth_depth, mask, camera.FoVx, camera.FoVy, rescale_depth_gt)
                o3d.io.write_point_cloud(os.path.join(gt_path, '{0:05d}'.format(idx) + ".ply"), pcd_gt)
                tifffile.imwrite(os.path.join(gt_path, '{0:05d}'.format(idx) + "_depth.tiff"), invdepth_gt_rescale.cpu().numpy())
            else:
                pcd = build_pcd(rendering, depth, torch.ones_like(depth).bool(), camera.FoVx, camera.FoVy)
            o3d.io.write_point_cloud(os.path.join(render_path, '{0:05d}'.format(idx) + ".ply"), pcd)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--no_rescale_depth_gt", action="store_true")
    parser.add_argument("--save_depth_pcd", action="store_true")
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration))
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=args.sh_degree, source=args.source, device=args.device, trainable_camera=args.mode == "camera",
            load_ply=load_ply, load_camera=args.load_camera, load_depth=True)
        rendering(dataset, gaussians, save, save_pcd=args.save_depth_pcd, rescale_depth_gt=not args.no_rescale_depth_gt)
