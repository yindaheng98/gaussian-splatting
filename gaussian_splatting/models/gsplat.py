import math
import torch
from gsplat import rasterization

from gaussian_splatting import GaussianModel, Camera


class GsplatGaussianModel(GaussianModel):

    def __init__(self, sh_degree, render_mode="RGB+D"):
        super(GsplatGaussianModel, self).__init__(sh_degree)
        self.render_mode = render_mode

    def forward(self, viewpoint_camera: Camera):
        """Rasterization using gsplat backend. Adapted from gsplat/examples/simple_viewer.py"""

        width = int(viewpoint_camera.image_width)
        height = int(viewpoint_camera.image_height)
        device = self._xyz.device

        # Construct viewmats [1, 4, 4] — undo Inria's transpose convention
        viewmats = viewpoint_camera.world_view_transform.T[None]  # [1, 4, 4]

        # Construct Ks [1, 3, 3] from FoV
        fx = width / (2 * math.tan(viewpoint_camera.FoVx * 0.5))
        fy = height / (2 * math.tan(viewpoint_camera.FoVy * 0.5))
        Ks = torch.tensor(
            [[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]],
            device=device,
        )[None]  # [1, 3, 3]

        # Rasterize — copied from gsplat/examples/simple_viewer.py
        render_colors, render_alphas, info = rasterization(
            self.get_xyz,                    # [N, 3]
            self.get_rotation,               # [N, 4]
            self.get_scaling,                # [N, 3]
            self.get_opacity.squeeze(-1),    # [N]
            self.get_features,               # [N, K, 3]
            viewmats,                        # [1, 4, 4]
            Ks,                              # [1, 3, 3]
            width,
            height,
            sh_degree=self.active_sh_degree,
            render_mode=self.render_mode,
            packed=False,
            rasterize_mode="antialiased" if self.antialiasing else "classic",
            backgrounds=viewpoint_camera.bg_color[None],  # [1, 3]
        )
        # render_colors: [1, H, W, 4], render_alphas: [1, H, W, 1]

        # Convert gsplat [1, H, W, C] output to Inria [C, H, W] convention
        rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1)  # [3, H, W]
        depth_image = render_colors[0, ..., 3:4].permute(2, 0, 1)     # [1, H, W]

        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)
        rendered_image = rendered_image.clamp(0, 1)

        # gsplat radii shape: [C, N, 2] (x and y pixel radii), Inria radii shape: [N]
        radii = info["radii"][0].max(dim=-1).values  # [1, N, 2] -> [N]

        out = {
            # capable for Inria GaussianModel
            "render": rendered_image,
            "viewspace_points": info["means2d"].squeeze(0),  # TODO: is this correct?
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "depth": 1 / depth_image,  # Inria depth is inverse depth, gsplat depth is accumulated depth
            # original gsplat output
            "info": info,
        }
        return out
