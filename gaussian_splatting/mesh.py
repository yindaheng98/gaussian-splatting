import os
from argparse import ArgumentParser
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import backends
from gaussian_splatting.render import prepare_rendering
from gaussian_splatting.utils import fov2focal


def post_process_mesh(mesh: o3d.geometry.TriangleMesh, n_cluster_to_keep: int, min_cluster_triangles: int) -> o3d.geometry.TriangleMesh:
    clusters, sizes, _ = mesh.cluster_connected_triangles()
    clusters, sizes = np.asarray(clusters), np.asarray(sizes)
    if not len(sizes):
        return mesh
    n_cluster_to_keep = min(n_cluster_to_keep, len(sizes))
    mesh.remove_triangles_by_mask(sizes[clusters] < max(np.sort(sizes)[-n_cluster_to_keep], min_cluster_triangles))
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return mesh


@torch.no_grad()
def extract_mesh(
        dataset: CameraDataset, gaussians: GaussianModel, save: str,
        depth_trunc_scale: float = 2.0, voxel_size_scale: float = 1.0, sdf_trunc_scale: float = 5.0, mesh_res: int = 1024,
        n_cluster_to_keep: int = 50, min_cluster_triangles: int = 50) -> None:
    gaussians.active_sh_degree = 0
    scene_extent = dataset.scene_extent()
    depth_trunc = depth_trunc_scale * scene_extent
    voxel_size = voxel_size_scale * depth_trunc / mesh_res
    sdf_trunc = sdf_trunc_scale * voxel_size
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in tqdm(range(len(dataset)), desc="TSDF integration"):
        camera = dataset[i]
        out = gaussians(camera)
        if "render" not in out or "depth" not in out:
            raise KeyError("Mesh extraction requires render and depth outputs")
        rgb = out["render"].cpu()
        depth = torch.nan_to_num(out["depth"].squeeze(0), nan=0, posinf=0, neginf=0).cpu()
        mask = camera.ground_truth_image_mask
        if mask is not None:
            depth[mask.cpu() < 0.5] = 0
        width, height = camera.image_width, camera.image_height
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fov2focal(camera.FoVx, width), fov2focal(camera.FoVy, height),
            (width - 1) / 2, (height - 1) / 2,
        )
        color = o3d.geometry.Image(np.asarray(
            rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255, order="C", dtype=np.uint8
        ))
        depth = o3d.geometry.Image(np.asarray(depth.numpy(), order="C", dtype=np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        volume.integrate(rgbd, intrinsic, camera.world_view_transform.T.cpu().numpy())
    os.makedirs(save, exist_ok=True)
    mesh = volume.extract_triangle_mesh()
    raw_path, post_path = os.path.join(save, "fuse.ply"), os.path.join(save, "fuse_post.ply")
    o3d.io.write_triangle_mesh(raw_path, mesh)
    o3d.io.write_triangle_mesh(post_path, post_process_mesh(mesh, n_cluster_to_keep, min_cluster_triangles))
    print(f"mesh saved at {raw_path}\npost-processed mesh saved at {post_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--backend", choices=backends, default="inria")
    parser.add_argument("-s", "--source", required=True, type=str)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_camera", default=None, type=str)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--no_image_mask", action="store_true")
    parser.add_argument("-o", "--option", default=[], action="append", type=str)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply")
    dataset, gaussians = prepare_rendering(
        sh_degree=args.sh_degree, source=args.source, device=args.device,
        trainable_camera=args.mode == "camera", load_ply=load_ply, load_camera=args.load_camera,
        load_mask=not args.no_image_mask, load_depth=False, backend=args.backend,
    )
    configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
    extract_mesh(dataset, gaussians, os.path.join(args.destination, f"ours_{args.iteration}"), **configs)
