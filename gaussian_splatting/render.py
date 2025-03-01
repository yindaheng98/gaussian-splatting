from typing import Tuple
import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, JSONCameraDataset, TrainableCameraDataset
from gaussian_splatting.dataset.colmap import ColmapCameraDataset, ColmapTrainableCameraDataset
from gaussian_splatting.utils import psnr
from gaussian_splatting.utils.lpipsPyTorch import lpips


def prepare_rendering(sh_degree: int, source: str, device: str, mode: str, load_ply: str, load_camera: str = None) -> Tuple[CameraDataset, GaussianModel]:
    match mode:
        case "base" | "densify":
            gaussians = GaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply)
            dataset = (JSONCameraDataset(load_camera) if load_camera else ColmapCameraDataset(source)).to(device)
        case "camera" | "camera-densify":
            gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply)
            dataset = (TrainableCameraDataset.from_json(load_camera) if load_camera else ColmapTrainableCameraDataset(source)).to(device)
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return dataset, gaussians


def rendering(dataset: CameraDataset, gaussians: GaussianModel, save: str):
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
            load_ply=load_ply, load_camera=args.load_camera)
        rendering(dataset, gaussians, save)
