import random
from tqdm import tqdm
from argparse import ArgumentParser
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset.colmap import ColmapCameraDataset, colmap_init
from gaussian_splatting.trainer import DensificationTrainer

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", default=30000, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--densify_from_iter", default=500, type=int)
parser.add_argument("--densify_until_iter", default=15000, type=int)
parser.add_argument("--densification_interval", default=100, type=int)
parser.add_argument("--opacity_reset_interval", default=3000, type=int)


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    gaussians = GaussianModel(sh_degree, device=device)
    dataset = ColmapCameraDataset(source, device=device)
    spatial_lr_scale = colmap_init(gaussians, args.source, dataset)
    trainer = DensificationTrainer(
        gaussians,
        spatial_lr_scale=spatial_lr_scale,
        densify_from_iter=args.densify_from_iter,
        densify_until_iter=args.densify_until_iter,
        densification_interval=args.densification_interval,
        opacity_reset_interval=args.opacity_reset_interval,
    )

    pbar = tqdm(range(iteration))
    epoch, epoch_loss = list(range(len(dataset))), []
    for step in pbar:
        epoch_idx = step % len(dataset)
        if epoch_idx == 0:
            random.shuffle(epoch)
            pbar.set_postfix({'epoch': step // len(dataset), 'loss': sum(epoch_loss) / len(dataset)})
            epoch_loss = []
        idx = epoch[epoch_idx]
        loss, out, gt = trainer.step(dataset[idx])
        epoch_loss.append(loss.item())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
