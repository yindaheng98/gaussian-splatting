import math
import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement

from gaussian_splatting.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.simple_knn._C import distCUDA2

from .utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation, RGB2SH
from .camera import Camera


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class GaussianModel(nn.Module):

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree):
        super(GaussianModel, self).__init__()
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()
        self.scale_modifier = 1.0
        self.debug = False
        self.antialiasing = False

    def to(self, device):
        self._xyz = nn.Parameter(self._xyz.to(device))
        self._features_dc = nn.Parameter(self._features_dc.to(device))
        self._features_rest = nn.Parameter(self._features_rest.to(device))
        self._opacity = nn.Parameter(self._opacity.to(device))
        self._scaling = nn.Parameter(self._scaling.to(device))
        self._rotation = nn.Parameter(self._rotation.to(device))
        return self

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

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

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=viewpoint_camera.bg_color.to(self._xyz.device),
            scale_modifier=self.scale_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug,
            antialiasing=self.antialiasing
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = self.get_xyz
        means2D = screenspace_points
        opacity = self.get_opacity

        scales = self.get_scaling
        rotations = self.get_rotation

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

    def create_from_pcd(self, points: torch.Tensor, colors: torch.Tensor):
        assert points.shape[0] == colors.shape[0]
        assert points.shape[1] == colors.shape[1] == 3
        device = self._xyz.device
        points, colors = points.to(dtype=torch.float, device=device), colors.to(dtype=torch.float, device=device)
        fused_point_cloud = points
        fused_color = RGB2SH(colors)
        features = torch.zeros((points.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device=device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), dtype=torch.float, device=device)
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        return self

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path: str):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path: str):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        device = self._xyz.device
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))

    def update_points_add(
            self,
            xyz: nn.Parameter,
            features_dc: nn.Parameter,
            features_rest: nn.Parameter,
            scaling: nn.Parameter,
            rotation: nn.Parameter,
            opacity: nn.Parameter,
    ):
        def is_same_prefix(attr: nn.Parameter, ref: nn.Parameter):
            return (attr[:ref.shape[0]] == ref).all()
        assert is_same_prefix(xyz, self._xyz)
        assert is_same_prefix(features_dc, self._features_dc)
        assert is_same_prefix(features_rest, self._features_rest)
        assert is_same_prefix(scaling, self._scaling)
        assert is_same_prefix(rotation, self._rotation)
        assert is_same_prefix(opacity, self._opacity)
        self._xyz = xyz
        self._features_dc = features_dc
        self._features_rest = features_rest
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity

    def update_points_remove(
            self, removed_mask: torch.Tensor,
            xyz: nn.Parameter,
            features_dc: nn.Parameter,
            features_rest: nn.Parameter,
            scaling: nn.Parameter,
            rotation: nn.Parameter,
            opacity: nn.Parameter,
    ):
        def is_same_rest(attr: nn.Parameter, ref: nn.Parameter):
            return (attr == ref[~removed_mask, ...]).all()
        assert is_same_rest(xyz, self._xyz)
        assert is_same_rest(features_dc, self._features_dc)
        assert is_same_rest(features_rest, self._features_rest)
        assert is_same_rest(scaling, self._scaling)
        assert is_same_rest(rotation, self._rotation)
        assert is_same_rest(opacity, self._opacity)
        self._xyz = xyz
        self._features_dc = features_dc
        self._features_rest = features_rest
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
