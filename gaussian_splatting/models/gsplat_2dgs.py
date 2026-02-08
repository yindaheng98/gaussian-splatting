import math
import torch
from gsplat.rendering import rasterization_2dgs

from gaussian_splatting import GaussianModel, Camera


class Gsplat2DGSGaussianModel(GaussianModel):

    def __init__(self, sh_degree, render_mode="RGB+D"):
        super(Gsplat2DGSGaussianModel, self).__init__(sh_degree)
        self.render_mode = render_mode

    def forward(self, viewpoint_camera: Camera):
        """Rasterization using gsplat 2DGS backend.

        Adapted from gsplat/examples/simple_trainer_2dgs.py and
        gsplat/gsplat/rendering.py::rasterization_2dgs.
        """

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

        # Rasterize using 2DGS — rasterization_2dgs returns 7 values
        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            self.get_xyz,                    # [N, 3]
            self._rotation,                  # [N, 4] raw quats — gsplat normalizes internally
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
            backgrounds=viewpoint_camera.bg_color[None],  # [1, 3]
        )
        # render_colors: [1, H, W, 4] (RGB+D), render_alphas: [1, H, W, 1]
        # render_normals: [1, H, W, 3]  — always present from rasterize_to_pixels_2dgs
        # normals_from_depth: [H, W, 3] — present when render_mode in ["RGB+D", "RGB+ED"]
        #                                  (.squeeze(0) reduces [1,H,W,3] to [H,W,3] since C=1)
        # render_distort: [1, H, W, 1], render_median: [1, H, W, 1]

        # Convert gsplat [1, H, W, C] output to Inria [C, H, W] convention
        rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1)  # [3, H, W]
        depth_image = render_colors[0, ..., 3:4].permute(2, 0, 1)     # [1, H, W]

        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)
        rendered_image = rendered_image.clamp(0, 1)

        # gsplat radii shape: [C, N, 2] (x and y pixel radii), Inria radii shape: [N]
        radii = info["radii"][0].max(dim=-1).values  # [1, N, 2] -> [N]

        # For 2DGS, the densification gradient is stored in info["gradient_2dgs"].
        # This is a tensor created inside rasterization_2dgs with requires_grad=True,
        # and participates in the backward pass for densification.
        #
        # Similar to the 3DGS gsplat backend (which uses info["means2d"]):
        # retain_grad() on the original tensor, then register_hook to forward the
        # gradient onto the squeezed viewspace_points for the Inria-style densifier.
        gradient_2dgs = info["gradient_2dgs"]  # [C, N, 2]
        gradient_2dgs.retain_grad()
        viewspace_points = gradient_2dgs.squeeze(0)  # [N, 2]
        gradient_2dgs.register_hook(lambda grad: setattr(viewspace_points, 'grad', grad.squeeze(0)))

        # Convert render_normals: [1, H, W, 3] -> [3, H, W]
        render_normals_out = render_normals[0].permute(2, 0, 1)  # [3, H, W]

        # Convert normals_from_depth: [H, W, 3] -> [3, H, W]
        # (rasterization_2dgs applies .squeeze(0) internally, so with C=1 this is [H, W, 3])
        normals_from_depth_out = normals_from_depth.permute(2, 0, 1)  # [3, H, W]

        out = {
            # compatible with Inria GaussianModel
            "render": rendered_image,
            "viewspace_points": viewspace_points,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "invdepth": 1 / depth_image,
            # 2DGS-specific outputs
            "depth": depth_image,
            "render_normals": render_normals_out,
            "normals_from_depth": normals_from_depth_out,
            "render_distort": render_distort,
            "render_median": render_median,
        }
        return out
