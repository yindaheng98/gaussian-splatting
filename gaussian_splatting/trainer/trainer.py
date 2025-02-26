from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch

from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.utils import l1_loss, ssim
from gaussian_splatting.utils.schedular import get_expon_lr_func


class AbstractTrainer(ABC):

    @property
    @abstractmethod
    def curr_step(self) -> int:
        raise ValueError("Current step is not set")

    @curr_step.setter
    @abstractmethod
    def curr_step(self, v):
        raise ValueError("Current step is not set")

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

    @abstractmethod
    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        pass

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
        out = self.model(camera)
        loss = self.loss(out, camera)
        loss.backward()
        self.before_optim_hook(loss=loss, out=out, camera=camera)
        self.optim_step()
        self.after_optim_hook(loss=loss, out=out, camera=camera)
        return loss, out

    def before_optim_hook(self, loss: torch.Tensor, out: dict, camera: Camera):
        pass

    def after_optim_hook(self, loss: torch.Tensor, out: dict, camera: Camera):
        pass


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
            rotation_lr=0.001,
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
        self._curr_step = 0

    @property
    def curr_step(self) -> int:
        return self._curr_step

    @curr_step.setter
    def curr_step(self, v):
        self._curr_step = v

    @property
    def model(self) -> GaussianModel:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def schedulers(self) -> Dict[str, Callable[[int], float]]:
        return self._schedulers

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        render = out["render"]
        gt = camera.ground_truth_image
        Ll1 = l1_loss(render, gt)
        ssim_value = ssim(render, gt)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim_value)
        return loss


class TrainerWrapper(AbstractTrainer):
    '''
    This class is designed to wrap a trainer and add additional functionality.
    Without this class, you should modify the trainer class directly.

    e.g.
    2 trainer in this package are inherited from this class: Densifier and CameraOptimizer.
    You can easily combine them to get a "DensifierCameraOptimizer" by Densifier(CameraOptimizer(AbstractTrainer(...), ...), ...),
    rather than define a new class "DensifierCameraOptimizer" that inherents from Densifier and CameraOptimizer.
    '''

    def __init__(self, base_trainer: AbstractTrainer):
        super().__init__()
        self.base_trainer = base_trainer

    # Implement the abstract methods of `AbstractTrainer`
    @property
    def curr_step(self) -> int:
        return self.base_trainer.curr_step

    @curr_step.setter
    def curr_step(self, v):
        self.base_trainer.curr_step = v

    @property
    def model(self) -> GaussianModel:
        return self.base_trainer.model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.base_trainer.optimizer

    @property
    def schedulers(self) -> Dict[str, Callable[[int], float]]:
        return self.base_trainer.schedulers

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        return self.base_trainer.loss(out, camera)

    # Override the methods of `AbstractTrainer`
    def update_learning_rate(self):
        return self.base_trainer.update_learning_rate()

    def optim_step(self):
        return self.base_trainer.optim_step()

    def before_optim_hook(self, loss: torch.Tensor, out: dict, camera: Camera):
        return self.base_trainer.before_optim_hook(loss=loss, out=out, camera=camera)

    def after_optim_hook(self, loss: torch.Tensor, out: dict, camera: Camera):
        return self.base_trainer.after_optim_hook(loss=loss, out=out, camera=camera)

    """
    The top-level methods `step` should call the overrided `loss`, `update_learning_rate` and `optim_step`.
    So do not override them to be `self.base_trainer.step`.
    """
