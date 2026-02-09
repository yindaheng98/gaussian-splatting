import torch
from .gaussian_model import GaussianModel, Camera
from .utils import normalize_quaternion, quaternion_to_matrix, quaternion_raw_multiply


class CameraTrainableGaussianModel(GaussianModel):

    def forward(self, viewpoint_camera: Camera):
        # means3D = pc.get_xyz
        rel_w2c = torch.eye(4, device=self._xyz.device)
        quaternion = viewpoint_camera.quaternion
        rel_w2c[:3, :3] = quaternion_to_matrix(normalize_quaternion(quaternion.unsqueeze(0))).squeeze(0)
        rel_w2c[:3, 3] = viewpoint_camera.T
        # Transform mean and rot of Gaussians to camera frame
        gaussians_xyz = self.get_xyz.clone()
        gaussians_rot = self.get_rotation.clone()

        xyz_ones = torch.ones(gaussians_xyz.shape[0], 1, dtype=gaussians_xyz.dtype, device=gaussians_xyz.device)
        xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
        gaussians_xyz_trans = (rel_w2c.detach().inverse() @ rel_w2c @ xyz_homo.T).T[:, :3]
        gaussians_rot_trans = quaternion_raw_multiply(quaternion.detach() * torch.tensor([1, -1, -1, -1], device=quaternion.device), quaternion_raw_multiply(quaternion, gaussians_rot))

        return self.render(
            viewpoint_camera=viewpoint_camera,
            means3D=gaussians_xyz_trans,
            opacity=self.get_opacity,
            scales=self.get_scaling,
            rotations=gaussians_rot_trans,
            shs=self.get_features,
        )
