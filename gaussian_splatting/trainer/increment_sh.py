
import torch

from gaussian_splatting import GaussianModel, Camera
from .trainer import AbstractTrainer, TrainerWrapper, BaseTrainer
from .camera_trainable import CameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetDensificationTrainer


class LiftSHTrainer(TrainerWrapper):
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


def LiftSHBaseTrainer(
        model: GaussianModel,
        spatial_lr_scale: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return LiftSHTrainer(
        BaseTrainer(model, spatial_lr_scale, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def LiftSHCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return LiftSHTrainer(
        CameraTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def LiftSHBaseDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return LiftSHTrainer(
        BaseDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def LiftSHOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return LiftSHTrainer(
        OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )
