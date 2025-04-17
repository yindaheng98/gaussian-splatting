
import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.utils.schedular import get_expon_lr_func
from .abc import AbstractTrainer, TrainerWrapper
from .base import BaseTrainer


class DepthTrainer(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            depth_l1_weight_init=1.0,
            depth_l1_weight_final=0.01,
            depth_l1_weight_max_steps=30_000,
    ):
        super().__init__(base_trainer)
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
        inv_depth = out["depth"]
        inv_depth_gt = camera.ground_truth_depth
        norm_depth = (inv_depth - inv_depth.mean()) / (inv_depth.std() + 1e-6)
        norm_depth_gt = (inv_depth_gt - inv_depth_gt.mean()) / (inv_depth_gt.std() + 1e-6)
        depth_l1 = torch.abs((norm_depth - norm_depth_gt)).mean()
        return loss + depth_l1 * self.depth_l1_weight


def BaseDepthTrainer(
    model: GaussianModel,
    scene_extent: float,
    depth_l1_weight_init=1.0,
    depth_l1_weight_final=0.01,
    depth_l1_weight_max_steps=30_000,
    *args, **kwargs
) -> DepthTrainer:
    return DepthTrainer(
        BaseTrainer(
            model=model,
            scene_extent=scene_extent,
            *args, **kwargs
        ),
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps
    )
