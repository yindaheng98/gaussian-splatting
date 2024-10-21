from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch
import torch.nn as nn

from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.utils import l1_loss, ssim
from gaussian_splatting.utils.schedular import get_expon_lr_func


class AbstractTrainer(ABC):
    def __init__(self):
        super().__init__()
        self.curr_step = 0

    @property
    @abstractmethod
    def model(self) -> GaussianModel:
        raise ValueError("Model is not set")

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        raise ValueError("Optimizer is not set")

    @property
    @abstractmethod
    def schedulers(self) -> Dict[str, Callable[[int], float]]:
        raise ValueError("Schedulers is not set")

    def oneupSHdegree(self):
        if self.model.active_sh_degree < self.model.max_sh_degree:
            self.model.active_sh_degree += 1

    @abstractmethod
    def loss(self, out: dict, gt) -> torch.Tensor:
        pass

    def forward_backward(self, camera: Camera):
        out = self.model(camera)
        gt = camera.ground_truth_image
        loss = self.loss(out, gt)
        loss.backward()
        return loss, out, gt

    def update_learning_rate(self):
        self.curr_step += 1
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.schedulers:
                param_group['lr'] = self.schedulers[param_group["name"]](self.curr_step)

    def optim_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, camera: Camera):
        self.update_learning_rate()
        loss, out, gt = self.forward_backward(camera)
        self.optim_step()
        return loss, out, gt


class BaseTrainer(AbstractTrainer):
    def __init__(
            self, model: GaussianModel,
            spatial_lr_scale: float,
            lambda_dssim=0.2,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=30_000,
            feature_lr=0.0025,
            opacity_lr=0.025,
            scaling_lr=0.005,
            rotation_lr=0.001
    ):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        params = [
            {'params': [model._xyz], 'lr': position_lr_init * spatial_lr_scale, "name": "xyz"},
            {'params': [model._features_dc], 'lr': feature_lr, "name": "f_dc"},
            {'params': [model._features_rest], 'lr': feature_lr / 20.0, "name": "f_rest"},
            {'params': [model._opacity], 'lr': opacity_lr, "name": "opacity"},
            {'params': [model._scaling], 'lr': scaling_lr, "name": "scaling"},
            {'params': [model._rotation], 'lr': rotation_lr, "name": "rotation"}
        ]
        optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        schedulers = {
            "xyz": get_expon_lr_func(
                lr_init=position_lr_init*spatial_lr_scale,
                lr_final=position_lr_final*spatial_lr_scale,
                lr_delay_mult=position_lr_delay_mult,
                max_steps=position_lr_max_steps,
            )
        }
        self._model = model
        self._optimizer = optimizer
        self._schedulers = schedulers

    @property
    def model(self) -> GaussianModel:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def schedulers(self) -> Dict[str, Callable[[int], float]]:
        return self._schedulers

    def loss(self, out: dict, gt):
        render = out["render"]
        Ll1 = l1_loss(render, gt)
        ssim_value = ssim(render, gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)
        return loss


class TrainerWrapper(AbstractTrainer):
    def __init__(self, base_trainer: AbstractTrainer):
        super().__init__()
        self.base_trainer = base_trainer

    @property
    def model(self) -> nn.Module:
        return self.base_trainer.model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.base_trainer.optimizer

    @property
    def schedulers(self) -> Dict[str, Callable[[int], float]]:
        return self.base_trainer.schedulers

    def loss(self, out: dict, gt):
        return self.base_trainer.loss(out, gt)
