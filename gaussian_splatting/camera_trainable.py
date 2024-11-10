import torch
import math
from .gaussian_model import GaussianModel, Camera
from .utils import normalize_quaternion, quaternion_to_matrix, quaternion_raw_multiply

from gaussian_splatting.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.simple_knn._C import distCUDA2


class CameraTrainableGaussianModel(GaussianModel):

    def forward(self, viewpoint_camera: Camera):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device=self._xyz.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
        w2c = torch.eye(4, device=self._xyz.device)
        projmatrix = (
            w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        campos = w2c.inverse()[3, :3]
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=viewpoint_camera.bg_color.to(self._xyz.device),
            scale_modifier=self.scale_modifier,
            # viewmatrix=viewpoint_camera.world_view_transform,
            # projmatrix=viewpoint_camera.full_proj_transform,
            viewmatrix=w2c,
            projmatrix=projmatrix,
            sh_degree=self.active_sh_degree,
            # campos=viewpoint_camera.camera_center,
            campos=campos,
            prefiltered=False,
            debug=self.debug,
            antialiasing=self.antialiasing
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # means3D = pc.get_xyz
        rel_w2c = torch.eye(4, device=self._xyz.device)
        quaternion = viewpoint_camera.quaternion
        rel_w2c[:3, :3] = quaternion_to_matrix(normalize_quaternion(quaternion.unsqueeze(0))).squeeze(0)
        rel_w2c[:3, 3] = viewpoint_camera.T
        # Transform mean and rot of Gaussians to camera frame
        gaussians_xyz = self._xyz.clone()
        gaussians_rot = self._rotation.clone()

        xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
        xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
        gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
        gaussians_rot_trans = quaternion_raw_multiply(quaternion, gaussians_rot)
        means3D = gaussians_xyz_trans
        means2D = screenspace_points
        opacity = self.get_opacity

        scales = self.get_scaling
        rotations = gaussians_rot_trans  # pc.get_rotation

        shs = self.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None)
        rendered_image = viewpoint_camera.postprocess(viewpoint_camera, rendered_image)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rendered_image = rendered_image.clamp(0, 1)
        out = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "depth": depth_image
        }
        return out
