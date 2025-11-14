from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from .camera_trainable import CameraTrainerWrapper, BaseCameraTrainer
from .densifier import BaseDensificationTrainer, AdaptiveDensificationTrainer
from .opacity_reset import OpacityResetTrainerWrapper
from .sh_lift import SHLifter, BaseSHLiftTrainer
from .depth import DepthTrainerWrapper, BaseDepthTrainer


# Camera trainer


def DepthCameraTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, scene_extent, dataset, *args, **kwargs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return OpacityResetTrainerWrapper(BaseDensificationTrainer, model, scene_extent, *args, **kwargs)


def BaseOpacityResetAdaptiveDensificationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return OpacityResetTrainerWrapper(AdaptiveDensificationTrainer, model, scene_extent, *args, **kwargs)


def DepthOpacityResetDensificationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(BaseOpacityResetDensificationTrainer, model, scene_extent, *args, **kwargs)


def DepthOpacityResetAdaptiveDensificationTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(BaseOpacityResetAdaptiveDensificationTrainer, model, scene_extent, *args, **kwargs)


def BaseOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: BaseOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )


def BaseOpacityResetAdaptiveDensificationCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: BaseOpacityResetAdaptiveDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )


def DepthOpacityResetDensificationCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )


def DepthOpacityResetAdaptiveDensificationCameraTrainer(model: CameraTrainableGaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthOpacityResetAdaptiveDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, scene_extent, dataset, *args, **kwargs
    )


# SHLift trainers


def DepthSHLiftTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs):
    return DepthTrainerWrapper(BaseSHLiftTrainer, model, scene_extent, *args, **kwargs)


def BaseSHLiftCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        BaseCameraTrainer(model, scene_extent, dataset, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        DepthCameraTrainer(model, scene_extent, dataset, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        DepthOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftOpacityResetAdaptiveDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        DepthOpacityResetAdaptiveDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def BaseSHLiftOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        BaseOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def BaseSHLiftOpacityResetAdaptiveDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        BaseOpacityResetAdaptiveDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftOpacityResetDensificationCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        DepthOpacityResetDensificationCameraTrainer(model, scene_extent, dataset, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftOpacityResetAdaptiveDensificationCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        DepthOpacityResetAdaptiveDensificationCameraTrainer(model, scene_extent, dataset, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def BaseSHLiftOpacityResetDensificationCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        BaseOpacityResetDensificationCameraTrainer(model, scene_extent, dataset, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def BaseSHLiftOpacityResetAdaptiveDensificationCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        BaseOpacityResetAdaptiveDensificationCameraTrainer(model, scene_extent, dataset, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


# Aliases for default trainers

Trainer = BaseDepthTrainer
CameraTrainer = DepthCameraTrainer
OpacityResetDensificationTrainer = DepthOpacityResetDensificationTrainer
OpacityResetAdaptiveDensificationTrainer = DepthOpacityResetAdaptiveDensificationTrainer
OpacityResetDensificationCameraTrainer = DepthOpacityResetDensificationCameraTrainer
OpacityResetAdaptiveDensificationCameraTrainer = DepthOpacityResetAdaptiveDensificationCameraTrainer
SHLiftTrainer = DepthSHLiftTrainer
SHLiftCameraTrainer = DepthSHLiftCameraTrainer
SHLiftOpacityResetDensificationTrainer = DepthSHLiftOpacityResetDensificationTrainer
SHLiftOpacityResetAdaptiveDensificationTrainer = DepthSHLiftOpacityResetAdaptiveDensificationTrainer
SHLiftOpacityResetDensificationCameraTrainer = DepthSHLiftOpacityResetDensificationCameraTrainer
SHLiftOpacityResetAdaptiveDensificationCameraTrainer = DepthSHLiftOpacityResetAdaptiveDensificationCameraTrainer
