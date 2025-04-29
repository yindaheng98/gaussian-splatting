from typing import Callable
import torch
from gaussian_splatting import GaussianModel

from .abc import AbstractDensifier, DensifierWrapper
from .trainer import DensificationTrainer


class OpacityPruner(DensifierWrapper):

    def __init__(
        self, base_densifier: AbstractDensifier,
        scene_extent,
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval=100,
        prune_screensize_threshold=20,
        prune_percent_too_big=1,
        prune_opacity_threshold=0.005,
    ):
        super().__init__(base_densifier)
        self.scene_extent = scene_extent
        self.prune_from_iter = prune_from_iter
        self.prune_until_iter = prune_until_iter
        self.prune_interval = prune_interval
        self.prune_screensize_threshold = prune_screensize_threshold
        self.prune_percent_too_big = prune_percent_too_big
        self.prune_opacity_threshold = prune_opacity_threshold

        self.max_radii2D = None

    def update_densification_stats(self, out):
        visibility_filter, radii = out["visibility_filter"], out["radii"]
        if self.max_radii2D is None:
            new_size = self.model.get_xyz.shape[0]
            self.max_radii2D = torch.zeros((new_size), device=self.model._xyz.device)
        self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])

    def prune(self) -> torch.Tensor:
        prune_mask = (self.model.get_opacity < self.prune_opacity_threshold).squeeze()
        big_points_vs = self.max_radii2D > self.prune_screensize_threshold
        big_points_ws = self.model.get_scaling.max(dim=1).values > self.prune_percent_too_big * self.scene_extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        return prune_mask

    def densify_and_prune(self, loss, out, camera, step: int):
        ret = super().densify_and_prune(loss, out, camera, step)
        if step <= self.prune_until_iter:
            self.update_densification_stats(out)
        if self.prune_from_iter <= step <= self.prune_until_iter and step % self.prune_interval == 0:
            ret = ret._replace(remove_mask=self.prune() if ret.remove_mask is None else torch.logical_or(ret.remove_mask, self.prune()))
        return ret

    def after_densify_and_prune_hook(self, loss, out, camera):
        self.max_radii2D = None
        return super().after_densify_and_prune_hook(loss, out, camera)


def OpacityPrunerTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval=100,
        prune_screensize_threshold=20,
        prune_percent_too_big=1,
        prune_opacity_threshold=0.005,
        **kwargs):
    densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = OpacityPruner(
        densifier,
        scene_extent,
        prune_from_iter=prune_from_iter,
        prune_until_iter=prune_until_iter,
        prune_interval=prune_interval,
        prune_screensize_threshold=prune_screensize_threshold,
        prune_percent_too_big=prune_percent_too_big,
        prune_opacity_threshold=prune_opacity_threshold,
    )
    return DensificationTrainer(
        model, scene_extent,
        densifier,
        *args, **kwargs
    )
