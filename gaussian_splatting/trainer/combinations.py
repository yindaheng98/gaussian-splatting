from functools import partial

from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from .camera_trainable import CameraTrainerWrapper, BaseCameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetTrainerWrapper
from .sh_lift import SHLiftTrainerWrapper, BaseSHLiftTrainer
from .depth import DepthTrainerWrapper, BaseDepthTrainer
from .normal import NormalTrainerWrapper


# Camera trainer


def DepthCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, dataset, **configs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return OpacityResetTrainerWrapper(BaseDensificationTrainer, model, dataset, **configs)


def DepthOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(BaseOpacityResetDensificationTrainer, model, dataset, **configs)


def BaseOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(BaseOpacityResetDensificationTrainer, model, dataset, **configs)


def DepthOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(DepthOpacityResetDensificationTrainer, model, dataset, **configs)


# SHLift trainers


def DepthSHLiftTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(BaseSHLiftTrainer, model, dataset, **configs)


def BaseSHLiftCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return SHLiftTrainerWrapper(BaseCameraTrainer, model, dataset, **configs)


def DepthSHLiftCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return SHLiftTrainerWrapper(DepthCameraTrainer, model, dataset, **configs)


def DepthSHLiftOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return SHLiftTrainerWrapper(DepthOpacityResetDensificationTrainer, model, dataset, **configs)


def BaseSHLiftOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return SHLiftTrainerWrapper(BaseOpacityResetDensificationTrainer, model, dataset, **configs)


def DepthSHLiftOpacityResetDensificationCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return SHLiftTrainerWrapper(DepthOpacityResetDensificationCameraTrainer, model, dataset, **configs)


def BaseSHLiftOpacityResetDensificationCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return SHLiftTrainerWrapper(BaseOpacityResetDensificationCameraTrainer, model, dataset, **configs)


# Aliases for default trainers
Trainer = BaseDepthTrainer
CameraTrainer = DepthCameraTrainer
OpacityResetDensificationTrainer = DepthOpacityResetDensificationTrainer
OpacityResetDensificationCameraTrainer = DepthOpacityResetDensificationCameraTrainer
SHLiftTrainer = DepthSHLiftTrainer
SHLiftCameraTrainer = DepthSHLiftCameraTrainer
SHLiftOpacityResetDensificationTrainer = DepthSHLiftOpacityResetDensificationTrainer
SHLiftOpacityResetDensificationCameraTrainer = DepthSHLiftOpacityResetDensificationCameraTrainer


# 2DGS normal consistency and depth distortion trainers
NormalTrainer = partial(NormalTrainerWrapper, Trainer)
NormalCameraTrainer = partial(NormalTrainerWrapper, CameraTrainer)
NormalOpacityResetDensificationTrainer = partial(NormalTrainerWrapper, OpacityResetDensificationTrainer)
NormalOpacityResetDensificationCameraTrainer = partial(NormalTrainerWrapper, OpacityResetDensificationCameraTrainer)
NormalSHLiftTrainer = partial(NormalTrainerWrapper, SHLiftTrainer)
NormalSHLiftCameraTrainer = partial(NormalTrainerWrapper, SHLiftCameraTrainer)
NormalSHLiftOpacityResetDensificationTrainer = partial(NormalTrainerWrapper, SHLiftOpacityResetDensificationTrainer)
NormalSHLiftOpacityResetDensificationCameraTrainer = partial(NormalTrainerWrapper, SHLiftOpacityResetDensificationCameraTrainer)


__all__ = [
    "BaseOpacityResetDensificationTrainer",
    "BaseOpacityResetDensificationCameraTrainer",
    "BaseSHLiftCameraTrainer",
    "BaseSHLiftOpacityResetDensificationTrainer",
    "BaseSHLiftOpacityResetDensificationCameraTrainer",

    "Trainer",
    "CameraTrainer",
    "OpacityResetDensificationTrainer",
    "OpacityResetDensificationCameraTrainer",
    "SHLiftTrainer",
    "SHLiftCameraTrainer",
    "SHLiftOpacityResetDensificationTrainer",
    "SHLiftOpacityResetDensificationCameraTrainer",

    "NormalTrainer",
    "NormalCameraTrainer",
    "NormalOpacityResetDensificationTrainer",
    "NormalOpacityResetDensificationCameraTrainer",
    "NormalSHLiftTrainer",
    "NormalSHLiftCameraTrainer",
    "NormalSHLiftOpacityResetDensificationTrainer",
    "NormalSHLiftOpacityResetDensificationCameraTrainer",
]
