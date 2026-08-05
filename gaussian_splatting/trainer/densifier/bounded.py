from functools import partial
from typing import Callable

import torch

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import AbstractDensifier, DensificationInstruct
from .trainer import DensificationTrainer
from .densifier import SplitCloneDensifier


class BoundedSplitCloneDensifier(SplitCloneDensifier):
    """Adjust the densify grad threshold to keep growth within target bounds."""

    def __init__(
        self,
        *args,
        densify_target_lower_bound=10000,
        densify_target_upper_bound=10000*1000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.densify_target_lower_bound = densify_target_lower_bound
        self.densify_target_upper_bound = densify_target_upper_bound

    def densify(self) -> DensificationInstruct:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        n_points = grads.shape[0]

        # Parent split always densifies too-big points, so only the remaining
        # points can be controlled by changing the gradient threshold.
        too_big_pts_mask = torch.max(self.model.get_scaling, dim=1).values > self.densify_percent_too_big*self.scene_extent
        n_too_big = too_big_pts_mask.sum().item()

        n_lower_select = max(0, self.densify_target_lower_bound - n_points - n_too_big)
        n_upper_select = max(0, self.densify_target_upper_bound - n_points - n_too_big)

        gradscore = torch.norm(grads, dim=-1)
        gradscore_rest = gradscore[~too_big_pts_mask]

        grad_threshold = self.densify_grad_threshold
        if gradscore_rest.numel() > 0:
            # Start from the normal threshold, then clamp the number of selected
            # points to the lower/upper gaps by converting the target count back
            # into a top-k gradient threshold.
            n_select = (gradscore_rest >= self.densify_grad_threshold).sum().item()
            n_select = max(n_select, n_lower_select)
            n_select = min(n_select, n_upper_select)
            n_select = min(n_select, gradscore_rest.numel())

            grad_threshold = torch.topk(gradscore_rest, n_select).values[-1].item() if n_select > 0 else float("inf")

        clone = super().densify_and_clone(grads, grad_threshold, self.scene_extent)
        split = super().densify_and_split(grads, grad_threshold, self.scene_extent)
        return clone.merge(split)


def BoundedSplitCloneDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_interval=100,
        densify_grad_threshold=0.0002,
        densify_percent_dense=0.01,
        densify_percent_too_big=0.8,
        densify_min_scale=1e-6,
        densify_limit_n=None,
        densify_target_lower_bound=10000,
        densify_target_upper_bound=10000*1000,
        **configs):
    return BoundedSplitCloneDensifier(
        base_densifier_constructor(model, dataset, *args, **configs),
        dataset,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big,
        densify_min_scale=densify_min_scale,
        densify_limit_n=densify_limit_n,
        densify_target_lower_bound=densify_target_lower_bound,
        densify_target_upper_bound=densify_target_upper_bound,
    )


def BoundedSplitCloneDensifierTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        **configs):
    return DensificationTrainer.from_densifier_constructor(
        partial(BoundedSplitCloneDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs
    )
