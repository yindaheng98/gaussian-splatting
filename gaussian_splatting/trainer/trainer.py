from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.utils import l1_loss, ssim, get_expon_lr_func


class AbstractTrainer(ABC):
    def __init__(self, model: GaussianModel, optimizer: torch.optim.Optimizer):
        super().__init__()
        self._model = model
        self._model.active_sh_degree = 0
        self._optimizer = optimizer

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def oneupSHdegree(self):
        if self._model.active_sh_degree < self._model.max_sh_degree:
            self._model.active_sh_degree += 1

    @abstractmethod
    def loss(self, out: dict, gt) -> torch.Tensor:
        pass

    def forward_backward(self, camera: Camera):
        out = self.model(camera)
        gt = camera.ground_truth_image
        loss = self.loss(out, gt)
        loss.backward()
        return loss, out, gt

    def optim_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, camera: Camera):
        loss, out, gt = self.forward_backward(camera)
        self.optim_step()
        return loss, out, gt


class BaseTrainer(AbstractTrainer):
    def __init__(
            self, model: GaussianModel,
            spatial_lr_scale: float,
            lambda_dssim=0.2,
            position_lr_init=0.00016,
            feature_lr=0.0025,
            opacity_lr=0.025,
            scaling_lr=0.005,
            rotation_lr=0.001
    ):
        self.lambda_dssim = lambda_dssim
        params = [
            {'params': [model._xyz], 'lr': position_lr_init * spatial_lr_scale, "name": "xyz"},
            {'params': [model._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [model._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [model._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [model._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [model._rotation], 'lr': rotation_lr, "name": "rotation"}
        ]
        super().__init__(model, torch.optim.Adam(params, lr=0.0, eps=1e-15))

    def loss(self, out: dict, gt):
        render = out["render"]
        Ll1 = l1_loss(render, gt)
        ssim_value = ssim(render, gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)
        return loss


class TrainerWrapper(AbstractTrainer):
    def __init__(self, base_trainer: AbstractTrainer):
        super().__init__(base_trainer.model, base_trainer.optimizer)
        self.base_trainer = base_trainer

    def loss(self, out: dict, gt):
        return self.base_trainer.loss(out, gt)


class LRScheduler(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            spatial_lr_scale: float,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=30_000
    ):
        super().__init__(base_trainer)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=position_lr_init*spatial_lr_scale,
            lr_final=position_lr_final*spatial_lr_scale,
            lr_delay_mult=position_lr_delay_mult,
            max_steps=position_lr_max_steps,
        )
        self.curr_step = 1

    def optim_step(self):
        self.update_learning_rate(self.curr_step)
        super().optim_step()
        self.curr_step += 1

    def update_learning_rate(self, step: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(step)
                param_group['lr'] = lr
                return lr


def LRScheduledTrainer(
        model: GaussianModel,
        scene_extent: float,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        *args, **kwargs):
    return LRScheduler(
        BaseTrainer(model, scene_extent, *args, **kwargs),
        scene_extent, position_lr_init, position_lr_final, position_lr_delay_mult, position_lr_max_steps
    )
