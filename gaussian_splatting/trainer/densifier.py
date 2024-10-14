import torch
import torch.nn as nn
from gaussian_splatting.utils import build_rotation
from gaussian_splatting.gaussian_model import GaussianModel


class Densifier:
    def __init__(self, model: GaussianModel, device="cuda"):
        self._xyz = model._xyz
        self._features_dc = model._features_dc
        self._features_rest = model._features_rest
        self._opacity = model._opacity
        self._scaling = model._scaling
        self._rotation = model._rotation
        self.xyz_gradient_accum = torch.zeros((model.get_xyz.shape[0], 1), device=device)
        self.denom = torch.zeros((model.get_xyz.shape[0], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=device)

    def update_max_radii2D(self, radii: torch.Tensor, visibility_filter: torch.Tensor):
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densification_prefix(self, radii, viewspace_points, visibility_filter):
        self.update_max_radii2D(radii, visibility_filter)
        self.add_densification_stats(viewspace_points, visibility_filter)
