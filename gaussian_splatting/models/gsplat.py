import torch
from gsplat import rasterization

from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.utils import normalize_quaternion, quaternion_to_matrix, quaternion_raw_multiply


class GsplatGaussianModel(GaussianModel):

    def __init__(self, sh_degree):
        super(GsplatGaussianModel, self).__init__(sh_degree)

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
        """Rasterization using gsplat backend. Adapted from gsplat/examples/simple_viewer.py"""

        width = int(viewpoint_camera.image_width)
        height = int(viewpoint_camera.image_height)
        device = means3D.device

        # Construct viewmats [1, 4, 4] — undo Inria's transpose convention
        viewmats = viewpoint_camera.world_view_transform.T[None]  # [1, 4, 4]

        Ks = viewpoint_camera.K.to(device=device, dtype=means3D.dtype)[None]  # [1, 3, 3]

        # Rasterize — copied from gsplat/examples/simple_viewer.py
        render_colors, render_alphas, info = rasterization(
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
            # RGB+ED returns alpha-normalized expected projection depth.
            # Should be more accurate than "RGB+D"
            render_mode="RGB+ED",
            packed=False,
            rasterize_mode="antialiased" if self.antialiasing else "classic",
            backgrounds=viewpoint_camera.bg_color[None],  # [1, 3]
        )
        # render_colors: [1, H, W, 4], render_alphas: [1, H, W, 1]

        # Convert gsplat [1, H, W, C] output to Inria [C, H, W] convention
        rendered_image = render_colors[0, ..., 0:3].permute(2, 0, 1)  # [3, H, W]
        # Match original PGSR: the zero rasterization background leaves
        # uncovered pixels at depth zero.
        depth_image = render_colors[0, ..., 3:4].permute(2, 0, 1)     # [1, H, W]

        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)
        rendered_image = rendered_image.clamp(0, 1)
        invdepth_image = self.inverse_depth(depth_image)

        # gsplat radii shape: [C, N, 2] (x and y pixel radii), Inria radii shape: [N]
        radii = info["radii"][0].max(dim=-1).values  # [1, N, 2] -> [N]

        # Capture means2d gradient for the Inria-style densifier.
        #
        # info["means2d"] (shape [C, N, 2]) is in the computation graph, so
        # retain_grad() reliably captures its gradient during backward().
        # We expose a get_viewspace_grad() accessor in `out` so the densifier
        # can read means2d.grad without hooks, closures, or reference cycles.
        means2d = info["means2d"]  # [C, N, 2]
        try:
            means2d.retain_grad()
        except:
            pass

        out = {
            # compatible with Inria GaussianModel
            "render": rendered_image,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "depth": depth_image,
            "invdepth": invdepth_image,
            # Used by the densifier to get the gradient of the viewspace points
            # gsplat's means2d gradient is in pixel space, but the Inria-style
            # densifier expects NDC-scale gradients (the original Inria rasterizer
            # backward produces gradients at a larger magnitude).  gsplat's own
            # DefaultStrategy multiplies by width/2 and height/2 to compensate
            # (see gsplat/strategy/default.py  _update_state).  We bake that
            # scaling into get_viewspace_grad so the densifier works unchanged.
            "get_viewspace_grad": lambda out: out["means2d"].grad.squeeze(0) * out["means2d"].new_tensor([[width, height]]) / 2.0,
            "means2d": means2d,
        }
        # Drop Python references to large rasterization intermediates.
        del render_colors, render_alphas, info
        return out


class CameraTrainableGsplatGaussianModel(GsplatGaussianModel):
    def forward(self, viewpoint_camera: Camera):
        # means3D = pc.get_xyz
        rel_w2c = torch.eye(4, device=self._xyz.device)
        quaternion = normalize_quaternion(viewpoint_camera.quaternion.unsqueeze(0)).squeeze(0)
        rel_w2c[:3, :3] = quaternion_to_matrix(quaternion)
        rel_w2c[:3, 3] = viewpoint_camera.T
        # Transform mean and rot of Gaussians to camera frame
        gaussians_xyz = self.get_xyz.clone()
        gaussians_rot = self.get_rotation.clone()

        xyz_ones = torch.ones(gaussians_xyz.shape[0], 1, dtype=gaussians_xyz.dtype, device=gaussians_xyz.device)
        xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
        gaussians_xyz_trans = (rel_w2c.detach().inverse() @ rel_w2c @ xyz_homo.T).T[:, :3]
        gaussians_rot_trans = quaternion_raw_multiply(quaternion.detach() * quaternion.new_tensor([1, -1, -1, -1]), quaternion_raw_multiply(quaternion, gaussians_rot))

        return self.render(
            viewpoint_camera=viewpoint_camera,
            means3D=gaussians_xyz_trans,
            opacity=self.get_opacity.squeeze(-1),
            scales=self.get_scaling,
            rotations=gaussians_rot_trans,
            shs=self.get_features,
        )
