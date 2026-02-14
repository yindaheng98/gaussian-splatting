from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import CameraDataset, TrainableCameraDataset
from .camera_trainable import CameraTrainerWrapper, BaseCameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetTrainerWrapper
from .sh_lift import SHLifter, BaseSHLiftTrainer
from .depth import DepthTrainerWrapper, BaseDepthTrainer


# Camera trainer


def DepthCameraTrainer(model: GaussianModel, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, dataset, **configs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return OpacityResetTrainerWrapper(BaseDensificationTrainer, model, dataset, **configs)


def DepthOpacityResetDensificationTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(BaseOpacityResetDensificationTrainer, model, dataset, **configs)


def BaseOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        lambda model, dataset, **configs: BaseOpacityResetDensificationTrainer(model, dataset, **configs),
        model, dataset, **configs
    )


def DepthOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        lambda model, dataset, **configs: DepthOpacityResetDensificationTrainer(model, dataset, **configs),
        model, dataset, **configs
    )


# SHLift trainers


def DepthSHLiftTrainer(model: GaussianModel, dataset: CameraDataset, **configs):
    return DepthTrainerWrapper(BaseSHLiftTrainer, model, dataset, **configs)


def BaseSHLiftCameraTrainer(
        model: GaussianModel,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BaseCameraTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftCameraTrainer(
        model: GaussianModel,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        DepthCameraTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftOpacityResetDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        DepthOpacityResetDensificationTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def BaseSHLiftOpacityResetDensificationTrainer(
        model: GaussianModel,
        dataset: CameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BaseOpacityResetDensificationTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftOpacityResetDensificationCameraTrainer(
        model: GaussianModel,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        DepthOpacityResetDensificationCameraTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def BaseSHLiftOpacityResetDensificationCameraTrainer(
        model: GaussianModel,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BaseOpacityResetDensificationCameraTrainer(model, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


# Aliases for default trainers
Trainer = BaseDepthTrainer
CameraTrainer = DepthCameraTrainer
OpacityResetDensificationTrainer = DepthOpacityResetDensificationTrainer
OpacityResetDensificationCameraTrainer = DepthOpacityResetDensificationCameraTrainer
SHLiftTrainer = DepthSHLiftTrainer
SHLiftCameraTrainer = DepthSHLiftCameraTrainer
SHLiftOpacityResetDensificationTrainer = DepthSHLiftOpacityResetDensificationTrainer
SHLiftOpacityResetDensificationCameraTrainer = DepthSHLiftOpacityResetDensificationCameraTrainer
