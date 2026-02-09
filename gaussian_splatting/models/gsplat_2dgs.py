import math
import torch
from gsplat.rendering import rasterization_2dgs

from gaussian_splatting import GaussianModel, Camera


class Gsplat2DGSGaussianModel(GaussianModel):

    def __init__(self, sh_degree, render_mode="RGB+D"):
        super(Gsplat2DGSGaussianModel, self).__init__(sh_degree)
        self.render_mode = render_mode

    def forward(self, viewpoint_camera: Camera):
        return self.render(
            viewpoint_camera=viewpoint_camera,
            means3D=self.get_xyz,
            opacity=self.get_opacity.squeeze(-1),
            scales=self.get_scaling,
            rotations=self._rotation,
            shs=self.get_features,
        )

    def render(
        self,
        viewpoint_camera: Camera,
        means3D: torch.Tensor,
        opacity: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        shs: torch.Tensor,
        colors_precomp=None,
        cov3D_precomp=None,
    ) -> dict:
        """Rasterization using gsplat 2DGS backend.

        Adapted from gsplat/examples/simple_trainer_2dgs.py and
        gsplat/gsplat/rendering.py::rasterization_2dgs.
        """

        width = int(viewpoint_camera.image_width)
        height = int(viewpoint_camera.image_height)
        device = means3D.device

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
            means3D,                         # [N, 3]
            rotations,                       # [N, 4] raw quats — gsplat normalizes internally
            scales,                          # [N, 3]
            opacity,             # [N]
            shs,                             # [N, K, 3]
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
        # retain_grad() reliably captures its gradient during backward().
        # We expose a get_viewspace_grad() accessor in `out` so the densifier
        # can read gradient_2dgs.grad without hooks, closures, or reference cycles.
        gradient_2dgs = info["gradient_2dgs"]  # [C, N, 2]
        gradient_2dgs.retain_grad()

        # gsplat's gradient_2dgs gradient is in pixel space, but the
        # Inria-style densifier expects NDC-scale gradients.  gsplat's own
        # DefaultStrategy multiplies by width/2 and height/2 to compensate
        # (see gsplat/strategy/default.py  _update_state).  We bake that
        # scaling into get_viewspace_grad so the densifier works unchanged.
        _grad_scale = gradient_2dgs.new_tensor([[width / 2.0, height / 2.0]])

        out = {
            # compatible with Inria GaussianModel
            "render": rendered_image,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "invdepth": 1 / depth_image,
            # Used by the densifier to get the gradient of the viewspace points
            "get_viewspace_grad": lambda out: out["gradient_2dgs"].grad.squeeze(0) * _grad_scale,
            "gradient_2dgs": gradient_2dgs,
        }
        # Explicitly free the large intermediate tensors from gsplat 2DGS rasterization.
        # (render_normals, normals_from_depth, render_distort, render_median are not
        # used by the current trainer/loss; keeping them in `out` wastes ~70 MB on GPU.)
        del render_colors, render_alphas, render_normals, normals_from_depth
        del render_distort, render_median, info
        return out
