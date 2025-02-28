
import torch

from gaussian_splatting import GaussianModel, Camera
from .abc import AbstractTrainer, TrainerWrapper
from .base import BaseTrainer
from .camera_trainable import BaseCameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetDensificationTrainer


class SHLiftTrainer(TrainerWrapper):
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


def SHLiftBaseTrainer(
        model: GaussianModel,
        spatial_lr_scale: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLiftTrainer(
        BaseTrainer(model, spatial_lr_scale, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def SHLiftCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLiftTrainer(
        BaseCameraTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def SHLiftBaseDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLiftTrainer(
        BaseDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def SHLiftOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLiftTrainer(
        OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )
