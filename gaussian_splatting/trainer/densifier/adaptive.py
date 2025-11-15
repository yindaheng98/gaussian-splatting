from typing import Callable
import torch
from gaussian_splatting.utils import build_rotation
from gaussian_splatting import GaussianModel

from .abc import AbstractDensifier, DensificationInstruct
from .trainer import DensificationTrainer
from .densifier import SplitCloneDensifier


class AdaptiveSplitCloneDensifier(SplitCloneDensifier):

    def __init__(
        self,
        *args,
        densify_target_n=10000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.densify_target_n = densify_target_n

    def densify_and_split(self, selected_pts_mask, scene_extent, N=2):
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_dense*scene_extent)
        # N=selected_pts_mask.sum(), add 2N new points and remove N old points

        stds = self.model.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.model._xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.model.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.model.scaling_inverse_activation(self.model.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self.model._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self.model._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self.model._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self.model._opacity[selected_pts_mask].repeat(N, 1)

        return DensificationInstruct(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacities=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
            remove_mask=selected_pts_mask
        )

    def densify_and_clone(self, selected_pts_mask, scene_extent):
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.get_scaling, dim=1).values <= self.densify_percent_dense*scene_extent)
        # N=selected_pts_mask.sum(), add N new points

        new_xyz = self.model._xyz[selected_pts_mask]
        new_features_dc = self.model._features_dc[selected_pts_mask]
        new_features_rest = self.model._features_rest[selected_pts_mask]
        new_opacities = self.model._opacity[selected_pts_mask]
        new_scaling = self.model._scaling[selected_pts_mask]
        new_rotation = self.model._rotation[selected_pts_mask]

        return DensificationInstruct(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacities=new_opacities,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
        )

    def densify(self) -> DensificationInstruct:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        too_big_pts_mask = torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_too_big*self.scene_extent
        n_should_select = max(0, self.densify_target_n - grads.shape[0] - too_big_pts_mask.sum().item())
        gradscore = torch.norm(grads, dim=-1)
        gradscore_rest = gradscore[~too_big_pts_mask]
        _, indices = torch.sort(gradscore_rest, descending=True)
        grad_threshold = gradscore_rest[indices[min(n_should_select, gradscore_rest.shape[0]) - 1]].item()
        if n_should_select <= 0:
            grad_threshold = self.densify_grad_threshold
        big_grad_pts_mask = gradscore >= min(grad_threshold, self.densify_grad_threshold)
        pts_mask = torch.logical_or(too_big_pts_mask, big_grad_pts_mask)

        clone = self.densify_and_clone(pts_mask, self.scene_extent)
        split = self.densify_and_split(pts_mask, self.scene_extent)

        return DensificationInstruct(
            new_xyz=torch.cat((clone.new_xyz, split.new_xyz), dim=0),
            new_features_dc=torch.cat((clone.new_features_dc, split.new_features_dc), dim=0),
            new_features_rest=torch.cat((clone.new_features_rest, split.new_features_rest), dim=0),
            new_opacities=torch.cat((clone.new_opacities, split.new_opacities), dim=0),
            new_scaling=torch.cat((clone.new_scaling, split.new_scaling), dim=0),
            new_rotation=torch.cat((clone.new_rotation, split.new_rotation), dim=0),
            remove_mask=split.remove_mask
        )


def AdaptiveSplitCloneDensifierTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_interval=100,
        densify_grad_threshold=0.0002,
        densify_percent_dense=0.01,
        densify_percent_too_big=0.8,
        densify_target_n=10000,
        **kwargs):
    densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = AdaptiveSplitCloneDensifier(
        densifier,
        scene_extent,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big,
        densify_target_n=densify_target_n,
    )
    return DensificationTrainer(
        model, scene_extent,
        densifier,
        *args, **kwargs
    )
