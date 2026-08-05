import torch
from gsplat.rendering import rasterization_2dgs

from gaussian_splatting import GaussianModel, Camera
from .gsplat import CameraTrainableGsplatGaussianModel


class Gsplat2DGSGaussianModel(GaussianModel):

    def __init__(self, sh_degree, min_scale=1e-6):
        super(Gsplat2DGSGaussianModel, self).__init__(sh_degree)
        self.min_scaling = self.scaling_inverse_activation(torch.tensor(min_scale)).item()

    def create_from_pcd(self, points: torch.Tensor, colors: torch.Tensor):
        super().create_from_pcd(points, colors)
        with torch.no_grad():
            self._scaling[:, 2] = self.min_scaling
        return self

    def load_ply(self, path: str):
        super().load_ply(path)
        with torch.no_grad():
            self._scaling[:, 2] = self.min_scaling

    def save_ply(self, path: str):
        with torch.no_grad():
            self._scaling[:, 2] = self.min_scaling
        super().save_ply(path)

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

        Ks = viewpoint_camera.K.to(device=device, dtype=means3D.dtype)[None]  # [1, 3, 3]

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
            render_mode="RGB+ED",
            packed=False,
            backgrounds=viewpoint_camera.bg_color[None],  # [1, 3]
            # Compute the per-pixel L1 pairwise depth-spread map
            # sum_ij(w_i * w_j * |z_i - z_j|).  This only produces
            # out["render_distort"]; the trainer must add it to the loss.
            distloss=True,
            # Use smooth, alpha-normalized expected depth for depth normals.
            depth_mode="expected",
        )
        # render_colors: [1, H, W, 4] (RGB+depth), render_alphas: [1, H, W, 1]
        # render_normals: [1, H, W, 3]  — always present from rasterize_to_pixels_2dgs
        # normals_from_depth: [H, W, 3]
        #                                  (.squeeze(0) reduces [1,H,W,3] to [H,W,3] since C=1)
        # render_distort: [1, H, W, 1], render_median: [1, H, W, 1]

        # Convert gsplat [1, H, W, C] output to Inria [C, H, W] convention
        rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1)  # [3, H, W]
        render_alphas_out = render_alphas[0].permute(2, 0, 1)         # [1, H, W]
        # Expected depth is the default surface; matching original PGSR, its
        # zero rasterization background leaves uncovered pixels at depth zero.
        depth_image = render_colors[0, ..., 3:4].permute(2, 0, 1)  # [1, H, W]

        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)
        rendered_image = rendered_image.clamp(0, 1)

        # gsplat radii shape: [C, N, 2] (x and y pixel radii), Inria radii shape: [N]
        radii = info["radii"][0].max(dim=-1).values  # [1, N, 2] -> [N]

        # Convert render_normals: [1, H, W, 3] -> [3, H, W]
        render_normals_out = render_normals[0].permute(2, 0, 1)  # [3, H, W]

        # Convert normals_from_depth: [H, W, 3] -> [3, H, W]
        # (rasterization_2dgs applies .squeeze(0) internally, so with C=1 this is [H, W, 3])
        normals_from_depth_out = normals_from_depth.permute(2, 0, 1)  # [3, H, W]

        # For 2DGS, the densification gradient is stored in info["gradient_2dgs"].
        # retain_grad() reliably captures its gradient during backward().
        # We expose a get_viewspace_grad() accessor in `out` so the densifier
        # can read gradient_2dgs.grad without hooks, closures, or reference cycles.
        gradient_2dgs = info["gradient_2dgs"]  # [C, N, 2]
        try:
            gradient_2dgs.retain_grad()
        except:
            pass

        out = {
            # compatible with Inria GaussianModel
            "render": rendered_image,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "depth": depth_image,
            # Direct reciprocal intentionally maps uncovered depth zero to Inf.
            "invdepth": 1 / depth_image,
            # Used by the densifier to get the gradient of the viewspace points
            # gsplat's gradient_2dgs gradient is in pixel space, but the
            # Inria-style densifier expects NDC-scale gradients.  gsplat's own
            # DefaultStrategy multiplies by width/2 and height/2 to compensate
            # (see gsplat/strategy/default.py  _update_state).  We bake that
            # scaling into get_viewspace_grad so the densifier works unchanged.
            "get_viewspace_grad": lambda out: out["gradient_2dgs"].grad.squeeze(0) * out["gradient_2dgs"].new_tensor([[width, height]]) / 2.0,
            "gradient_2dgs": gradient_2dgs,
            # Additional outputs from 2DGS (normals and distortion)
            "render_normals": render_normals_out,
            "normals_from_depth": normals_from_depth_out,
            "render_alphas": render_alphas_out,
            "render_distort": render_distort,
            "render_median": render_median,
        }
        # Explicitly free only intermediate containers. The geometry outputs
        # remain referenced by `out` for normal consistency and depth distortion.
        del render_colors, render_alphas, info
        return out


class CameraTrainableGsplat2DGSGaussianModel(Gsplat2DGSGaussianModel):
    def forward(self, viewpoint_camera: Camera):
        return CameraTrainableGsplatGaussianModel.forward(self, viewpoint_camera)
