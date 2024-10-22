import json
import os
import random
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.utils import psnr
from gaussian_splatting.dataset.colmap import ColmapCameraDataset, colmap_init, ColmapTrainableCameraDataset, colmap_compute_scene_extent
from gaussian_splatting.trainer import BaseTrainer, DensificationTrainer, CameraTrainer

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", default=30000, type=int)
parser.add_argument("-l", "--load_ply", default=None, type=str)
parser.add_argument("--mode", choices=["pure", "densify", "camera"], default="pure")
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--config", default=None, type=str)


def read_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def init_gaussians(sh_degree: int, source: str, device: str, mode: str, load_ply: str = None, configs={}):
    def init(gaussians: GaussianModel, source: str, load_ply: str = None):
        if load_ply:
            gaussians.load_ply(load_ply)
        else:
            colmap_init(gaussians, source)

    match mode:
        case "pure":
            dataset = ColmapCameraDataset(source).to(device)
            gaussians = GaussianModel(sh_degree).to(device)
            init(gaussians, source, load_ply)
            trainer = BaseTrainer(
                gaussians,
                spatial_lr_scale=colmap_compute_scene_extent(dataset),
                **configs
            )
        case "densify":
            dataset = ColmapCameraDataset(source).to(device)
            gaussians = GaussianModel(sh_degree).to(device)
            init(gaussians, source, load_ply)
            trainer = DensificationTrainer(
                gaussians,
                scene_extent=colmap_compute_scene_extent(dataset),
                **configs
            )
        case "camera":
            dataset = ColmapTrainableCameraDataset(source).to(device)
            gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
            init(gaussians, source, load_ply)
            trainer = CameraTrainer(
                gaussians,
                scene_extent=colmap_compute_scene_extent(dataset),
                dataset=dataset,
                **configs
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    if load_ply:
        gaussians.activate_all_sh_degree()
    return dataset, gaussians, trainer


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    configs = {} if args.config is None else read_config(args.config)
    dataset, gaussians, trainer = init_gaussians(sh_degree, source, device, args.mode, args.load_ply, configs)
    dataset.save_cameras(os.path.join(destination, "cameras.json"))

    pbar = tqdm(range(1, iteration+1))
    epoch = list(range(len(dataset)))
    epoch_psnr = torch.empty(3, 0, device=device)
    ema_loss_for_log = 0.0
    avg_psnr_for_log = 0.0
    for step in pbar:
        epoch_idx = step % len(dataset)
        if epoch_idx == 0:
            avg_psnr_for_log = epoch_psnr.mean().item()
            epoch_psnr = torch.empty(3, 0, device=device)
            random.shuffle(epoch)
        idx = epoch[epoch_idx]
        loss, out, gt = trainer.step(dataset[idx])
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            epoch_psnr = torch.concat([epoch_psnr, psnr(out["render"], gt)], dim=1)
            if step % 10 == 0:
                pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log, 'psnr': avg_psnr_for_log, 'n': gaussians._xyz.shape[0]})
        if step in args.save_iterations:
            save_path = os.path.join(destination, "point_cloud", "iteration_" + str(step))
            os.makedirs(save_path, exist_ok=True)
            gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
    save_path = os.path.join(destination, "point_cloud", "iteration_" + str(iteration))
    os.makedirs(save_path, exist_ok=True)
    gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))


if __name__ == "__main__":
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(False)
    main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
