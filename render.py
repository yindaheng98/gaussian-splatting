import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset.colmap import ColmapCameraDataset

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", required=True, type=int)
parser.add_argument("--device", default="cuda", type=str)


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str):
    gaussians = GaussianModel(sh_degree).to(device)
    gaussians.load_ply(os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))
    dataset = ColmapCameraDataset(source).to(device)
    render_path = os.path.join(destination, "ours_{}".format(iteration), "renders")
    gt_path = os.path.join(destination, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    for idx, camera in enumerate(tqdm(dataset, desc="Rendering progress")):
        out = gaussians(camera)
        rendering = out["render"]
        gt = camera.ground_truth_image
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        main(args.sh_degree, args.source, args.destination, args.iteration, args.device)
