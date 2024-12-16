import json
import os
import random
import torch
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import JSONCameraDataset, TrainableCameraDataset
from gaussian_splatting.utils import psnr
from gaussian_splatting.dataset.colmap import ColmapCameraDataset, colmap_init, ColmapTrainableCameraDataset
from gaussian_splatting.trainer import BaseTrainer, DensificationTrainer, CameraTrainer

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", default=30000, type=int)
parser.add_argument("-l", "--load_ply", default=None, type=str)
parser.add_argument("--load_camera", default=None, type=str)
parser.add_argument("--mode", choices=["pure", "densify", "camera"], default="pure")
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("-o", "--option", default=[], action='append', type=str)


def prepare_training(sh_degree: int, source: str, device: str, mode: str, load_ply: str = None, load_camera: str = None, configs={}):
    match mode:
        case "pure":
            gaussians = GaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
            dataset = (JSONCameraDataset(load_camera) if load_camera else ColmapCameraDataset(source)).to(device)
            trainer = BaseTrainer(
                gaussians,
                spatial_lr_scale=dataset.scene_extent(),
                **configs
            )
        case "densify":
            gaussians = GaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
            dataset = (JSONCameraDataset(load_camera) if load_camera else ColmapCameraDataset(source)).to(device)
            trainer = DensificationTrainer(
                gaussians,
                scene_extent=dataset.scene_extent(),
                **configs
            )
        case "camera":
            gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply) if load_ply else colmap_init(gaussians, source)
            dataset = (TrainableCameraDataset.from_json(load_camera) if load_camera else ColmapTrainableCameraDataset(source)).to(device)
            trainer = CameraTrainer(
                gaussians,
                scene_extent=dataset.scene_extent(),
                dataset=dataset,
                **configs
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    if load_ply:
        gaussians.activate_all_sh_degree()
    return dataset, gaussians, trainer


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    with open(os.path.join(destination, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    dataset, gaussians, trainer = prepare_training(
        sh_degree=sh_degree, source=source, device=device, mode=args.mode,
        load_ply=args.load_ply, load_camera=args.load_camera, configs=configs)
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
        loss, out = trainer.step(dataset[idx])
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            epoch_psnr = torch.concat([epoch_psnr, psnr(out["render"], dataset[idx].ground_truth_image)], dim=1)
            if step % 10 == 0:
                pbar.set_postfix({'epoch': step // len(dataset), 'loss': ema_loss_for_log, 'psnr': avg_psnr_for_log, 'n': gaussians._xyz.shape[0]})
        if step in args.save_iterations:
            save_path = os.path.join(destination, "point_cloud", "iteration_" + str(step))
            os.makedirs(save_path, exist_ok=True)
            gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
            dataset.save_cameras(os.path.join(destination, "cameras.json"))
    save_path = os.path.join(destination, "point_cloud", "iteration_" + str(iteration))
    os.makedirs(save_path, exist_ok=True)
    gaussians.save_ply(os.path.join(save_path, "point_cloud.ply"))
    dataset.save_cameras(os.path.join(destination, "cameras.json"))


if __name__ == "__main__":
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(False)
    main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
