
from typing import Callable

import torch

from gaussian_splatting import GaussianModel, Camera
from gaussian_splatting.dataset import CameraDataset
from .base import BaseTrainer
from .abc import AbstractTrainer, TrainerWrapper
from .registry import trainer_wrap


class SHLifter(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            sh_degree_up_interval=1000,
            initial_sh_degree=0,
    ):
        super().__init__(base_trainer)
        self.sh_degree_up_interval = sh_degree_up_interval
        self.model.active_sh_degree = initial_sh_degree

    def oneupSHdegree(self):
        if self.model.active_sh_degree < self.model.max_sh_degree:
            self.model.active_sh_degree += 1

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if self.curr_step % self.sh_degree_up_interval == 0:
            self.oneupSHdegree()
        return super().loss(out, camera)


@trainer_wrap("shlift")
def SHLiftTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        base_trainer_constructor(model, dataset, *args, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree,
    )


def BaseSHLiftTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return SHLiftTrainerWrapper(BaseTrainer, model, dataset, **configs)
