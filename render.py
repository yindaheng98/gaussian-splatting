import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser, Namespace
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.dataset.colmap import ColmapTrainableCameraDataset
from gaussian_splatting.utils import psnr
from lpipsPyTorch import lpips
from gaussian_splatting.dataset import JSONCameraDataset
from gaussian_splatting.dataset.colmap import ColmapCameraDataset

parser = ArgumentParser()
parser.add_argument("--sh_degree", default=3, type=int)
parser.add_argument("-s", "--source", required=True, type=str)
parser.add_argument("-d", "--destination", required=True, type=str)
parser.add_argument("-i", "--iteration", required=True, type=int)
parser.add_argument("--load_camera", default=None, type=str)
parser.add_argument("--mode", choices=["pure", "densify", "camera"], default="pure")
parser.add_argument("--device", default="cuda", type=str)


def init_gaussians(sh_degree: int, source: str, device: str, mode: str, load_ply: str, load_camera: str = None):
    match mode:
        case "pure" | "densify":
            gaussians = GaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply)
            dataset = (JSONCameraDataset(load_camera) if load_camera else ColmapCameraDataset(source)).to(device)
        case "camera":
            gaussians = CameraTrainableGaussianModel(sh_degree).to(device)
            gaussians.load_ply(load_ply)
            dataset = (ColmapTrainableCameraDataset(source) if load_camera else TrainableCameraDataset.from_json(load_camera)).to(device)
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return dataset, gaussians


def transform2d_pixel(H, W, device="cuda"):
    x = torch.arange(W, dtype=torch.float, device=device)
    y = torch.arange(H, dtype=torch.float, device=device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
    A = torch.rand((2, 2)).to(device) - 0.5
    b = (torch.rand(2).to(device) - 0.5) * H
    solution = torch.cat([b[:, None], A], dim=1).T
    xy_transformed = (xy.view(-1, 2) @ A.T + b).view(xy.shape)
    # X = torch.cat([torch.ones((xy.view(-1, 2).shape[0], 1)).to(device=xy.device), xy.view(-1, 2)], dim=1)
    # Y = xy_transformed.view(-1, 2)
    # diff = solution - torch.linalg.lstsq(X, Y).solution
    return xy_transformed, solution


def solve_sigma(T, cov2D):
    X = torch.zeros((T.shape[0], 3, 6), device=T.device)
    # 1st row for x
    X[..., 0, 0] = T[..., 0, 0] ** 2
    X[..., 0, 1] = 2 * T[..., 0, 1] * T[..., 0, 0]
    X[..., 0, 2] = 2 * T[..., 0, 2] * T[..., 0, 0]
    X[..., 0, 3] = T[..., 0, 1] ** 2
    X[..., 0, 4] = 2 * T[..., 0, 1] * T[..., 0, 2]
    X[..., 0, 5] = T[..., 0, 2] ** 2
    # 2nd row for y
    X[..., 1, 0] = T[..., 1, 0] * T[..., 0, 0]
    X[..., 1, 1] = T[..., 1, 1] * T[..., 0, 0] + T[..., 1, 0] * T[..., 0, 1]
    X[..., 1, 2] = T[..., 1, 2] * T[..., 0, 0] + T[..., 1, 0] * T[..., 0, 2]
    X[..., 1, 3] = T[..., 1, 1] * T[..., 0, 1]
    X[..., 1, 4] = T[..., 1, 1] * T[..., 0, 2] + T[..., 1, 2] * T[..., 0, 1]
    X[..., 1, 5] = T[..., 1, 2] * T[..., 0, 2]
    # 3rd row for z
    X[..., 2, 0] = T[..., 1, 0] ** 2
    X[..., 2, 1] = 2 * T[..., 1, 1] * T[..., 1, 0]
    X[..., 2, 2] = 2 * T[..., 1, 2] * T[..., 1, 0]
    X[..., 2, 3] = T[..., 1, 1] ** 2
    X[..., 2, 4] = 2 * T[..., 1, 1] * T[..., 1, 2]
    X[..., 2, 5] = T[..., 1, 2] ** 2
    # solve underdetermined system of equations
    Y = torch.zeros((T.shape[0], 3, 1), device=T.device)
    Y[..., 0, 0] = cov2D[..., 0, 0]  # for x
    Y[..., 1, 0] = cov2D[..., 0, 1]  # for y
    Y[..., 2, 0] = cov2D[..., 1, 1]  # for z
    return X, Y


def main(sh_degree: int, source: str, destination: str, iteration: int, device: str, args):
    with open(os.path.join(destination, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(sh_degree=sh_degree, source_path=source)))
    dataset, gaussians = init_gaussians(
        sh_degree=sh_degree, source=source, device=device, mode=args.mode,
        load_ply=os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"),
        load_camera=args.load_camera)
    render_path = os.path.join(destination, "ours_{}".format(iteration), "renders")
    gt_path = os.path.join(destination, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gt_path, exist_ok=True)
    pbar = tqdm(dataset, desc="Rendering progress")
    for idx, camera in enumerate(pbar):
        xy_transformed, solution = transform2d_pixel(camera.image_height, camera.image_width, device=device)
        out = gaussians.motion_fusion(camera, xy_transformed)
        rendering = out["render"]
        gt = camera.ground_truth_image
        pbar.set_postfix({"PSNR": psnr(rendering, gt).mean().item(), "LPIPS": lpips(rendering, gt).mean().item()})
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, '{0:05d}'.format(idx) + ".png"))

        print("\nframe", idx)
        # verify exported data
        valid_idx = (out['radii'] > 0) & (out['tran_det'] > 1e-3)
        B = out['transform2d'][..., 0:6].reshape(-1, 2, 3)[valid_idx]
        eqs = out['transform2d'][..., 6:27].reshape(-1, 3, 7)[valid_idx]
        conv3D = out['transform2d'][..., 27:36].reshape(-1, 3, 3)[valid_idx]
        conv2D = out['transform2d'][..., 36:40].reshape(-1, 2, 2)[valid_idx]
        T = out['transform2d'][..., 40:49].reshape(-1, 3, 3)[valid_idx]
        print("T \Sigma_{3D} T^\\top - \Sigma_{2D}", (T.bmm(conv3D).bmm(T.transpose(1, 2))[:, :2, :2] - conv2D).abs().mean())
        A2D, b2D = B[..., :-1], B[..., -1]
        conv2D_transformed = torch.zeros((conv2D.shape[0], 2, 2), device=conv2D.device)
        conv2D_transformed[:, 0, 0] = eqs[..., 0, -1]
        conv2D_transformed[:, 0, 1] = eqs[..., 1, -1]
        conv2D_transformed[:, 1, 0] = eqs[..., 1, -1]
        conv2D_transformed[:, 1, 1] = eqs[..., 2, -1]
        print("A \Sigma_{2D} A^\\top - \Sigma'_{2D}", (A2D.bmm(conv2D).bmm(A2D.transpose(1, 2)) - conv2D_transformed).abs().mean())

        # solve underdetermined system of equations
        X, Y = eqs[..., :-1], eqs[..., -1].unsqueeze(-1)
        # X0, Y0 = X.clone(), Y.clone()
        # X, Y = solve_sigma(T, conv2D_transformed)
        # print((X0 - X).abs().mean(), (Y0 - Y).abs().mean())
        rank = torch.linalg.matrix_rank(X)
        valid_idx = (rank == 3)
        qr = torch.linalg.qr(X[valid_idx].transpose(1, 2))
        sigma_flatten = qr.Q.bmm(torch.linalg.inv(qr.R).transpose(1, 2)).bmm(Y[valid_idx]).squeeze(-1)
        print("A_{T} \Sigma_{3D} - b_{T}", (X.bmm(sigma_flatten.unsqueeze(-1)) - Y).abs().mean())

        sigma = torch.zeros((sigma_flatten.shape[0], 3, 3), device=sigma_flatten.device)
        sigma[:, 0, 0] = sigma_flatten[:, 0]
        sigma[:, 0, 1] = sigma_flatten[:, 1]
        sigma[:, 0, 2] = sigma_flatten[:, 2]
        sigma[:, 1, 0] = sigma_flatten[:, 1]
        sigma[:, 1, 1] = sigma_flatten[:, 3]
        sigma[:, 1, 2] = sigma_flatten[:, 4]
        sigma[:, 2, 0] = sigma_flatten[:, 2]
        sigma[:, 2, 1] = sigma_flatten[:, 4]
        sigma[:, 2, 2] = sigma_flatten[:, 5]

        # verify equations
        B = B[valid_idx]
        eqs = eqs[valid_idx]
        conv3D = conv3D[valid_idx]
        conv2D = conv2D[valid_idx]
        T = T[valid_idx]
        print("T \Sigma_{3D} T^\\top - \Sigma_{2D}", (T.bmm(conv3D).bmm(T.transpose(1, 2))[:, :2, :2] - conv2D).abs().mean())
        A2D, b2D = B[..., :-1], B[..., -1]
        conv2D_transformed = torch.zeros((conv2D.shape[0], 2, 2), device=conv2D.device)
        conv2D_transformed[:, 0, 0] = eqs[..., 0, -1]
        conv2D_transformed[:, 0, 1] = eqs[..., 1, -1]
        conv2D_transformed[:, 1, 0] = eqs[..., 1, -1]
        conv2D_transformed[:, 1, 1] = eqs[..., 2, -1]
        print("A \Sigma_{2D} A^\\top - \Sigma'_{2D}", (A2D.bmm(conv2D).bmm(A2D.transpose(1, 2)) - conv2D_transformed).abs().mean())
        print("T \Sigma'_{3D} T^\\top - \Sigma'_{2D}", (T.bmm(sigma).bmm(T.transpose(1, 2))[:, :2, :2] - conv2D_transformed).abs().mean())
        pass


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
