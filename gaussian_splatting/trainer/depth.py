
from typing import Callable
import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.utils.schedular import get_expon_lr_func
from .abc import AbstractTrainer, TrainerWrapper
from .base import BaseTrainer


class DepthTrainer(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            depth_from_iter=7500,
            depth_rescale_gt: bool = True,
            depth_rescale_gt_max=1.0,
            depth_l1_weight_init=1.0,
            depth_l1_weight_final=0.01,
            depth_l1_weight_max_steps=30_000,
    ):
        super().__init__(base_trainer)
        self.depth_from_iter = depth_from_iter
        self.depth_rescale_gt = depth_rescale_gt
        self.depth_rescale_gt_max = depth_rescale_gt_max
        self.depth_l1_weight_func = get_expon_lr_func(
            depth_l1_weight_init,
            depth_l1_weight_final,
            depth_l1_weight_max_steps)
        self.depth_l1_weight = self.depth_l1_weight_func(0)

    def update_learning_rate(self):
        super().update_learning_rate()
        self.depth_l1_weight = self.depth_l1_weight_func(self.curr_step)

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = super().loss(out, camera)
        if self.curr_step < self.depth_from_iter or camera.ground_truth_depth is None:
            return loss
        inv_depth = out["depth"].squeeze(0)
        inv_depth_gt = camera.ground_truth_depth
        if self.depth_rescale_gt:
            mask = inv_depth_gt > (1 / self.depth_rescale_gt_max)
            mean_gt, std_gt = inv_depth_gt[mask].mean(), inv_depth_gt[mask].std()
            mean, std = inv_depth[mask].mean(), inv_depth[mask].std()
        else:
            mean_gt, std_gt = inv_depth_gt.mean(), inv_depth_gt.std()
            mean, std = mean_gt, std_gt
        norm_depth = (inv_depth - mean) / std
        norm_depth_gt = (inv_depth_gt - mean_gt) / std_gt
        depth_dist = torch.abs(norm_depth - norm_depth_gt)
        if camera.ground_truth_depth_mask is not None:
            depth_dist *= camera.ground_truth_depth_mask
        depth_l1 = depth_dist.mean()
        return loss + depth_l1 * self.depth_l1_weight


# Depth is the one of the core components of the Gaussian Splatting
# but considering there are different methods for depth loss, implement this in BaseTrainer.loss is not a good idea
# DepthTrainerWrapper is used to wrap the base trainer with depth loss
def DepthTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        scene_extent: float,
        *args,
        depth_from_iter=7500,
        depth_rescale_gt: bool = True,
        depth_rescale_gt_max=1.0,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_l1_weight_max_steps=30_000,
        **kwargs) -> DepthTrainer:
    return DepthTrainer(
        base_trainer=base_trainer_constructor(model, scene_extent, *args, **kwargs),
        depth_from_iter=depth_from_iter,
        depth_rescale_gt=depth_rescale_gt,
        depth_rescale_gt_max=depth_rescale_gt_max,
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps
    )


def BaseDepthTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs) -> DepthTrainer:
    return DepthTrainerWrapper(BaseTrainer, model, scene_extent, *args, **kwargs)
