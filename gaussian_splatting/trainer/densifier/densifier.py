from typing import Callable
import torch
from gaussian_splatting.utils import build_rotation
from gaussian_splatting import GaussianModel

from .abc import AbstractDensifier, DensifierWrapper, DensificationInstruct
from .trainer import DensificationTrainer


class SplitCloneDensifier(DensifierWrapper):

    def __init__(
        self, base_densifier: AbstractDensifier,
        scene_extent,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_interval=100,
        densify_grad_threshold=0.0002,
        densify_percent_dense=0.01,
        densify_percent_too_big=0.8,
    ):
        super().__init__(base_densifier)
        self.scene_extent = scene_extent
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densify_interval = densify_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_percent_dense = densify_percent_dense
        self.densify_percent_too_big = densify_percent_too_big

        self.xyz_gradient_accum = None
        self.denom = None

    def update_densification_stats(self, out):
        viewspace_points, visibility_filter = out["viewspace_points"], out["visibility_filter"]
        if self.xyz_gradient_accum is None or self.denom is None:
            new_size = self.model.get_xyz.shape[0]
            self.xyz_gradient_accum = torch.zeros((new_size, 1), device=self.model._xyz.device)
            self.denom = torch.zeros((new_size, 1), device=self.model._xyz.device)
        self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_points.grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.model.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.model._xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_dense*scene_extent)
        selected_pts_mask = torch.logical_or(
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_too_big*scene_extent)

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

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.model.get_scaling, dim=1).values <= self.densify_percent_dense*scene_extent)

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

        clone = self.densify_and_clone(grads, self.densify_grad_threshold, self.scene_extent)
        split = self.densify_and_split(grads, self.densify_grad_threshold, self.scene_extent)

        return DensificationInstruct(
            new_xyz=torch.cat((clone.new_xyz, split.new_xyz), dim=0),
            new_features_dc=torch.cat((clone.new_features_dc, split.new_features_dc), dim=0),
            new_features_rest=torch.cat((clone.new_features_rest, split.new_features_rest), dim=0),
            new_opacities=torch.cat((clone.new_opacities, split.new_opacities), dim=0),
            new_scaling=torch.cat((clone.new_scaling, split.new_scaling), dim=0),
            new_rotation=torch.cat((clone.new_rotation, split.new_rotation), dim=0),
            remove_mask=split.remove_mask
        )

    def densify_and_prune(self, loss, out, camera, step: int):
        ret = super().densify_and_prune(loss, out, camera, step)
        if step <= self.densify_until_iter:
            self.update_densification_stats(out)
        if self.densify_from_iter <= step <= self.densify_until_iter and step % self.densify_interval == 0:
            dense = self.densify()
            if ret.new_xyz is not None:
                ret = ret._replace(
                    new_xyz=torch.cat((ret.new_xyz, dense.new_xyz), dim=0),
                    new_features_dc=torch.cat((ret.new_features_dc, dense.new_features_dc), dim=0),
                    new_features_rest=torch.cat((ret.new_features_rest, dense.new_features_rest), dim=0),
                    new_opacities=torch.cat((ret.new_opacities, dense.new_opacities), dim=0),
                    new_scaling=torch.cat((ret.new_scaling, dense.new_scaling), dim=0),
                    new_rotation=torch.cat((ret.new_rotation, dense.new_rotation), dim=0),
                )
            else:
                ret = ret._replace(
                    new_xyz=dense.new_xyz,
                    new_features_dc=dense.new_features_dc,
                    new_features_rest=dense.new_features_rest,
                    new_opacities=dense.new_opacities,
                    new_scaling=dense.new_scaling,
                    new_rotation=dense.new_rotation,
                )
            ret = ret._replace(remove_mask=dense.remove_mask if ret.remove_mask is None else torch.logical_or(ret.remove_mask, dense.remove_mask))
        return ret

    def after_densify_and_prune_hook(self, loss, out, camera):
        self.xyz_gradient_accum = None
        self.denom = None
        return super().after_densify_and_prune_hook(loss, out, camera)


def SplitCloneDensifierTrainerWrapper(
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
        **kwargs):
    densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = SplitCloneDensifier(
        densifier,
        scene_extent,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big
    )
    return DensificationTrainer(
        model, scene_extent,
        densifier,
        *args, **kwargs
    )
