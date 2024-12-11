import math
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
from gaussian_splatting.diff_gaussian_rasterization.motion_utils import solve_cov3D, compute_mean2D, compute_T, compute_Jacobian, compute_cov2D, transform_cov2D, unflatten_symmetry_3x3

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
    X = torch.cat([torch.ones((xy.view(-1, 2).shape[0], 1)).to(device=xy.device), xy.view(-1, 2)], dim=1)
    Y = xy_transformed.view(-1, 2)
    print(solution - torch.linalg.lstsq(X, Y).solution)
    B = (X.T@X).inverse()@X.T@Y
    print(B.T[:, 0] - b, B.T[:, 1:] - A)
    v11 = X.unsqueeze(-1).bmm(X.unsqueeze(-1).transpose(1, 2)).sum(dim=0)
    v12 = X.unsqueeze(-1).bmm(Y.unsqueeze(-1).transpose(1, 2)).sum(dim=0)
    print((v11 - X.T@X) / v11, (v12 - X.T@Y) / v12)
    randidx = torch.randint(0, X.shape[0], size=(30,), device=X.device)
    print(X[randidx].unsqueeze(-1).bmm(X[randidx].unsqueeze(-1).transpose(1, 2)).sum(dim=0) - X[randidx].T@X[randidx])
    return xy_transformed, solution


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
        valid_idx = (out['radii'] > 0) & (out['motion_det'] > 1e-12) & (out['motion_alpha'] > 1e-3) & (out['pixhit'] > 1)
        motion_det = out['motion_det'][valid_idx]
        motion_alpha = out['motion_alpha'][valid_idx]
        pixhit = out['pixhit'][valid_idx]
        # verify exported data
        B = out['motion2d'].reshape(-1)[:out['motion2d'].shape[0] * 6].reshape(out['motion2d'].shape[0], 2, 3)[valid_idx]
        # T = out['motion2d'][..., 6:15].reshape(-1, 3, 3)[valid_idx]
        # conv3D0 = out['motion2d'][..., 6:12][valid_idx]
        conv3D = gaussians.get_covariance()[valid_idx]
        # print("conv3D", (conv3D - conv3D0).abs().max())
        J = compute_Jacobian(gaussians.get_xyz.detach(), camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.world_view_transform)
        T = compute_T(J, camera.world_view_transform)[valid_idx]
        # print("T", (T[:, :2, :] - T0[valid_idx]).abs().max())
        A2D, b2D = B[..., :-1], B[..., -1]

        # solve mean
        point_image = compute_mean2D(camera.full_proj_transform, camera.image_width, camera.image_height, gaussians.get_xyz.detach())
        point_image_ = out["mean2D"][:, 7:]
        print("point_image", (point_image[point_image_.abs().sum(1) > 0] - point_image[point_image_.abs().sum(1) > 0]).abs().mean())

        # solve cov2D
        conv2D = compute_cov2D(T, unflatten_symmetry_3x3(conv3D))
        conv2D_transformed = transform_cov2D(A2D, conv2D)

        # solve underdetermined system of equations
        X, Y = solve_cov3D(gaussians.get_xyz.detach()[valid_idx], camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.world_view_transform, conv2D_transformed)
        rank = torch.linalg.matrix_rank(X)
        valid_idx = (rank == 3)
        qr = torch.linalg.qr(X[valid_idx].transpose(1, 2))
        sigma_flatten = qr.Q.bmm(torch.linalg.inv(qr.R).transpose(1, 2)).bmm(Y[valid_idx]).squeeze(-1)
        print("A_{T} \Sigma_{3D} - b_{T}", (X.bmm(sigma_flatten.unsqueeze(-1)) - Y).abs().mean())  # !large value in Y will cause error in solving sigma

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

        motion_det = motion_det[valid_idx]
        motion_alpha = motion_alpha[valid_idx]
        pixhit = pixhit[valid_idx]
        # verify equations
        B = B[valid_idx]
        T = T[valid_idx]
        A2D, b2D = B[..., :-1], B[..., -1]
        conv2D_transformed = conv2D_transformed[valid_idx]
        print("T \Sigma'_{3D} T^\\top - \Sigma'_{2D}", (T.bmm(sigma).bmm(T.transpose(1, 2))[:, :2, :2] - conv2D_transformed).abs().mean())
        pass


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        main(args.sh_degree, args.source, args.destination, args.iteration, args.device, args)
