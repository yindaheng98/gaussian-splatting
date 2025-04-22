
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
            depth_rescale_mode: str = 'local',
            depth_local_relative_kernel_radius=8,
            depth_local_relative_stride=4,
            depth_l1_weight_init=1.0,
            depth_l1_weight_final=0.01,
            depth_l1_weight_max_steps=30_000,
    ):
        super().__init__(base_trainer)
        self.depth_from_iter = depth_from_iter
        self.depth_rescale_mode = depth_rescale_mode
        self.depth_local_relative_kernel_radius = depth_local_relative_kernel_radius
        self.depth_local_relative_stride = depth_local_relative_stride
        self.depth_l1_weight_func = get_expon_lr_func(
            depth_l1_weight_init,
            depth_l1_weight_final,
            depth_l1_weight_max_steps)
        self.depth_l1_weight = self.depth_l1_weight_func(0)

    def update_learning_rate(self):
        super().update_learning_rate()
        self.depth_l1_weight = self.depth_l1_weight_func(self.curr_step)

    def compute_global_relative_depth_loss(self, inv_depth: torch.Tensor, inv_depth_gt: torch.Tensor, mask: torch.Tensor = None, rescale_gt: bool = True):
        mean_gt, std_gt = inv_depth_gt.mean(), inv_depth_gt.std()
        mean, std = mean_gt, std_gt
        if rescale_gt:
            mean_gt, std_gt = inv_depth_gt.mean(), inv_depth_gt.std()
            mean, std = inv_depth.mean(), inv_depth.std()
        norm_depth = (inv_depth - mean) / std
        norm_depth_gt = (inv_depth_gt - mean_gt) / std_gt
        depth_dist = torch.abs(norm_depth - norm_depth_gt)
        if mask is not None:
            depth_dist *= mask
        return depth_dist.mean()

    def compute_local_relative_depth_loss(self, inv_depth: torch.Tensor, inv_depth_gt: torch.Tensor, mask: torch.Tensor = None):
        kernel_size = self.depth_local_relative_kernel_radius
        stride = self.depth_local_relative_stride
        height, width = inv_depth.shape
        kernel = torch.cartesian_prod(
            torch.arange(-kernel_size, kernel_size+1, dtype=torch.int64, device=inv_depth.device),
            torch.arange(-kernel_size, kernel_size+1, dtype=torch.int64, device=inv_depth.device))
        centers = torch.cartesian_prod(
            torch.arange(kernel_size, height - kernel_size, stride, dtype=torch.int64, device=inv_depth.device),
            torch.arange(kernel_size, width - kernel_size, stride, dtype=torch.int64, device=inv_depth.device))
        pix_idx = centers.unsqueeze(1) + kernel.unsqueeze(0)
        local_inv_depth = inv_depth[pix_idx[:, :, 0], pix_idx[:, :, 1]]
        local_inv_depth_gt = inv_depth_gt[pix_idx[:, :, 0], pix_idx[:, :, 1]]
        local_center_inv_depth = local_inv_depth[:, kernel.shape[0] // 2].unsqueeze(-1)
        local_center_inv_depth_gt = local_inv_depth_gt[:, kernel.shape[0] // 2].unsqueeze(-1)
        local_scale_inv_depth = local_inv_depth.std(-1).unsqueeze(-1)
        local_scale_inv_depth_gt = local_inv_depth_gt.std(-1).unsqueeze(-1)
        local_inv_depth_gt_rescaled = (local_inv_depth_gt - local_center_inv_depth_gt) / local_scale_inv_depth_gt * local_scale_inv_depth + local_center_inv_depth
        local_loss = local_inv_depth - local_inv_depth_gt_rescaled
        if mask is not None:
            local_loss *= mask[pix_idx[:, :, 0], pix_idx[:, :, 1]]
        return local_loss.abs().mean()

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = super().loss(out, camera)
        if self.curr_step < self.depth_from_iter or camera.ground_truth_depth is None:
            return loss
        inv_depth = out["depth"].squeeze(0)
        inv_depth_gt = camera.ground_truth_depth
        match(self.depth_rescale_mode):
            case 'local':
                depth_l1 = self.compute_local_relative_depth_loss(inv_depth, inv_depth_gt, camera.ground_truth_depth_mask)
            case 'global':
                depth_l1 = self.compute_global_relative_depth_loss(inv_depth, inv_depth_gt, camera.ground_truth_depth_mask, rescale_gt=True)
            case 'none':
                depth_l1 = self.compute_global_relative_depth_loss(inv_depth, inv_depth_gt, camera.ground_truth_depth_mask, rescale_gt=False)
            case _:
                raise ValueError(f"Unknown depth rescale mode: {self.depth_rescale_mode}")
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
        depth_rescale_mode: str = 'local',
        depth_local_relative_kernel_radius=8,
        depth_local_relative_stride=4,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_l1_weight_max_steps=30_000,
        **kwargs) -> DepthTrainer:
    return DepthTrainer(
        base_trainer=base_trainer_constructor(model, scene_extent, *args, **kwargs),
        depth_from_iter=depth_from_iter,
        depth_rescale_mode=depth_rescale_mode,
        depth_local_relative_kernel_radius=depth_local_relative_kernel_radius,
        depth_local_relative_stride=depth_local_relative_stride,
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps
    )


def BaseDepthTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs) -> DepthTrainer:
    return DepthTrainerWrapper(BaseTrainer, model, scene_extent, *args, **kwargs)
