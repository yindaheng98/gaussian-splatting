from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from .camera_trainable import CameraTrainerWrapper, BaseCameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetTrainerWrapper
from .sh_lift import SHLifter, BaseSHLiftTrainer
from .depth import DepthTrainerWrapper, BaseDepthTrainer


# Camera trainer


def DepthCameraTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, **configs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, scene_extent, dataset, **configs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(model: GaussianModel, scene_extent: float, **configs):
    return OpacityResetTrainerWrapper(BaseDensificationTrainer, model, scene_extent, **configs)


def DepthOpacityResetDensificationTrainer(model: GaussianModel, scene_extent: float, **configs):
    return DepthTrainerWrapper(BaseOpacityResetDensificationTrainer, model, scene_extent, **configs)


def BaseOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, **configs: BaseOpacityResetDensificationTrainer(model, scene_extent, **configs),
        model, scene_extent, dataset, **configs
    )


def DepthOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: TrainableCameraDataset, **configs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, **configs: DepthOpacityResetDensificationTrainer(model, scene_extent, **configs),
        model, scene_extent, dataset, **configs
    )


# SHLift trainers


def DepthSHLiftTrainer(model: GaussianModel, scene_extent: float, **configs):
    return DepthTrainerWrapper(BaseSHLiftTrainer, model, scene_extent, **configs)


def BaseSHLiftCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BaseCameraTrainer(model, scene_extent, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        DepthCameraTrainer(model, scene_extent, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        DepthOpacityResetDensificationTrainer(model, scene_extent, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def BaseSHLiftOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BaseOpacityResetDensificationTrainer(model, scene_extent, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftOpacityResetDensificationCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        DepthOpacityResetDensificationCameraTrainer(model, scene_extent, dataset, **configs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def BaseSHLiftOpacityResetDensificationCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        **configs):
    return SHLifter(
        BaseOpacityResetDensificationCameraTrainer(model, scene_extent, dataset, **configs),
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
