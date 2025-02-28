from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch

from gaussian_splatting import GaussianModel, Camera


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
