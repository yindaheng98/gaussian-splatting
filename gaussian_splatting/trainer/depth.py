
from typing import Callable
import torch
import torch.nn.functional as F

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.utils.schedular import get_expon_lr_func
from .abc import AbstractTrainer, TrainerWrapper
from .base import BaseTrainer


class DepthTrainer(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            depth_from_iter=7500,
            depth_resize: int = None,  # set this value to resize the depth map
            depth_rescale_mode: str = 'local',
            depth_local_relative_kernel_radius=8,
            depth_local_relative_stride=4,
            depth_l1_weight_init=1.0,
            depth_l1_weight_final=0.01,
            depth_l1_weight_max_steps=30_000,
    ):
        super().__init__(base_trainer)
        self.depth_from_iter = depth_from_iter
        self.depth_resize = depth_resize
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

    def compute_global_relative_depth_loss(self, invdepth: torch.Tensor, invdepth_gt: torch.Tensor, mask: torch.Tensor = None, rescale_gt: bool = True):
        mean_gt, std_gt = invdepth_gt.mean(), invdepth_gt.std()
        mean, std = mean_gt, std_gt
        if rescale_gt:
            mean_gt, std_gt = invdepth_gt.mean(), invdepth_gt.std()
            mean, std = invdepth.mean(), invdepth.std()
        norm_depth = (invdepth - mean) / std
        norm_depth_gt = (invdepth_gt - mean_gt) / std_gt
        depth_dist = torch.abs(norm_depth - norm_depth_gt)
        if mask is not None:
            depth_dist *= mask
        return depth_dist.mean()

    def compute_local_relative_depth_loss(self, invdepth: torch.Tensor, invdepth_gt: torch.Tensor, mask: torch.Tensor = None):
        kernel_size = self.depth_local_relative_kernel_radius*2 + 1
        stride = self.depth_local_relative_stride
        center_idx = kernel_size**2 // 2
        local_invdepth = F.unfold(invdepth.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=stride, padding=0).squeeze(0)
        local_invdepth_gt = F.unfold(invdepth_gt.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=stride, padding=0).squeeze(0)
        local_center_invdepth = local_invdepth[center_idx, :].unsqueeze(0).detach()
        local_center_invdepth_gt = local_invdepth_gt[center_idx, :].unsqueeze(0)
        local_scale_invdepth = local_invdepth.std(0).unsqueeze(0).detach()  # some region std may be 0, which will cause NaN in backward
        local_scale_invdepth_gt = local_invdepth_gt.std(0).unsqueeze(0)
        local_scale = local_scale_invdepth / local_scale_invdepth_gt
        local_scale[..., local_scale_invdepth_gt < 1e-6] = 1.0  # some region std may be 0, which will cause NaN in backward
        local_invdepth_gt_rescaled = (local_invdepth_gt - local_center_invdepth_gt) * local_scale + local_center_invdepth
        local_loss = local_invdepth - local_invdepth_gt_rescaled
        if mask is not None:
            local_loss *= F.unfold(mask.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=stride, padding=0).squeeze(0)
        return local_loss.abs().mean()

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        loss = super().loss(out, camera)
        if self.curr_step < self.depth_from_iter or camera.ground_truth_depth is None:
            return loss
        invdepth = out["depth"].squeeze(0)
        invdepth_gt = camera.ground_truth_depth
        mask = camera.ground_truth_depth_mask
        assert invdepth.shape == invdepth_gt.shape, f"invdepth shape {invdepth.shape} does not match gt depth shape {invdepth_gt.shape}"
        if self.depth_resize is not None:
            height, width = invdepth.shape[-2:]
            scale = self.depth_resize / max(height, width)
            height, width = int(height * scale), int(width * scale)
            invdepth = F.interpolate(invdepth.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            invdepth_gt = F.interpolate(invdepth_gt.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            if mask is not None:
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        match(self.depth_rescale_mode):
            case 'local':
                depth_l1 = self.compute_local_relative_depth_loss(invdepth, invdepth_gt, mask)
            case 'global':
                depth_l1 = self.compute_global_relative_depth_loss(invdepth, invdepth_gt, mask, rescale_gt=True)
            case 'none':
                depth_l1 = self.compute_global_relative_depth_loss(invdepth, invdepth_gt, mask, rescale_gt=False)
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
        depth_resize=None,
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
        depth_resize=depth_resize,
        depth_rescale_mode=depth_rescale_mode,
        depth_local_relative_kernel_radius=depth_local_relative_kernel_radius,
        depth_local_relative_stride=depth_local_relative_stride,
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps
    )


def BaseDepthTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs) -> DepthTrainer:
    return DepthTrainerWrapper(BaseTrainer, model, scene_extent, *args, **kwargs)
