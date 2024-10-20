import os
import random
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import psnr
from gaussian_splatting.dataset.colmap import ColmapCameraDataset, colmap_init
from gaussian_splatting.trainer import DensificationTrainer

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", default=30000, type=int)
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--densify_from_iter", default=500, type=int)
parser.add_argument("--densify_until_iter", default=15000, type=int)
parser.add_argument("--densification_interval", default=100, type=int)
parser.add_argument("--opacity_reset_interval", default=3000, type=int)


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    gaussians = GaussianModel(sh_degree, device=device)
    dataset = ColmapCameraDataset(source, device=device)
    scene_extent = colmap_init(gaussians, args.source, dataset)
    trainer = DensificationTrainer(
        gaussians,
        scene_extent=scene_extent,
        densify_from_iter=args.densify_from_iter,
        densify_until_iter=args.densify_until_iter,
        densification_interval=args.densification_interval,
        opacity_reset_interval=args.opacity_reset_interval,
    )

    pbar = tqdm(range(1, iteration+1))
    epoch, epoch_loss, epoch_psnr = list(range(len(dataset))), [], torch.empty(3, 0, device=device)
    for step in pbar:
        epoch_idx = step % len(dataset)
        if epoch_idx == 0:
            pbar.set_postfix({'epoch': step // len(dataset), 'loss': sum(epoch_loss) / len(dataset), 'psnr': epoch_psnr.mean().item()})
            epoch_loss, epoch_psnr = [], torch.empty(3, 0, device=device)
            random.shuffle(epoch)
        if step % 1000 == 0:
            trainer.oneupSHdegree()
        idx = epoch[epoch_idx]
        loss, out, gt = trainer.step(dataset[idx])
        with torch.no_grad():
            epoch_loss.append(loss.item())
            epoch_psnr = torch.concat([epoch_psnr, psnr(out["render"], gt)], dim=1)
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
