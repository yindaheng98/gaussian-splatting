import time
import os

import torch
import viser
import nerfview

from gaussian_splatting import GaussianModel, build_camera
from gaussian_splatting.utils import focal2fov
from gaussian_splatting.prepare import prepare_gaussians, backends


@torch.no_grad()
def viewer_render_fn(
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
        gaussians: GaussianModel, device: str,
        bg_color=(0., 0., 0.)):
    if render_tab_state.preview_render:
        width = render_tab_state.render_width
        height = render_tab_state.render_height
    else:
        width = render_tab_state.viewer_width
        height = render_tab_state.viewer_height

    c2w = camera_state.c2w  # [4, 4] numpy float64
    K = camera_state.get_K((width, height))  # [3, 3] numpy float64

    # Convert c2w to R, T (world-to-camera decomposition)
    c2w_torch = torch.from_numpy(c2w).float().to(device)
    w2c = torch.linalg.inv(c2w_torch)
    R = w2c[:3, :3]
    T = w2c[:3, 3]

    # Convert intrinsics to FoV
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)

    # Build Camera
    camera = build_camera(
        image_height=int(height), image_width=int(width),
        FoVx=FoVx, FoVy=FoVy,
        R=R, T=T,
        bg_color=bg_color, device=device,
    )
    print(f"Resolution: {width}x{height}")

    # Render
    return gaussians(camera)


def viewing(
        gaussians: GaussianModel, device: str,
        port: int = 8080, bg_color=(0., 0., 0.)) -> None:
    server = viser.ViserServer(port=port, verbose=False)
    viewer = nerfview.Viewer(
        server=server,
        render_fn=lambda cs, rts: viewer_render_fn(cs, rts, gaussians, device, bg_color)["render"].permute(1, 2, 0).cpu().numpy(),
        mode="rendering",
    )
    print(f"Viewer running on port {port}... Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--backend", choices=backends, default="inria")
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    with torch.no_grad():
        gaussians = prepare_gaussians(
            sh_degree=args.sh_degree, source=args.destination, device=args.device,
            load_ply=load_ply, backend=args.backend)
        viewing(gaussians, device=args.device, port=args.port)
