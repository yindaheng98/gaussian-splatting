from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from .camera_trainable import CameraTrainerWrapper, BaseCameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetTrainerWrapper
from .sh_lift import SHLifter, BaseSHLiftTrainer
from .depth import DepthTrainer, DepthTrainerWrapper, BaseDepthTrainer


# Camera trainer


def DepthCameraTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, scene_extent, dataset, *args, **kwargs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        *args, **kwargs):
    return OpacityResetTrainerWrapper(
        BaseDensificationTrainer,
        model, scene_extent,
        *args, **kwargs
    )


def DepthOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        depth_from_iter=7500,
        depth_resize=None,
        depth_rescale_mode: str = 'local',
        depth_local_relative_kernel_radius=4,
        depth_local_relative_stride=4,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_l1_weight_max_steps=30_000,
        *args, **kwargs):
    return DepthTrainer(
        BaseOpacityResetDensificationTrainer(
            model, scene_extent,
            *args, **kwargs
        ),
        depth_from_iter=depth_from_iter,
        depth_resize=depth_resize,
        depth_rescale_mode=depth_rescale_mode,
        depth_local_relative_kernel_radius=depth_local_relative_kernel_radius,
        depth_local_relative_stride=depth_local_relative_stride,
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps,
    )


def BaseOpacityResetDensificationCameraTrainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: BaseOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, dataset, scene_extent, *args, **kwargs
    )


def DepthOpacityResetDensificationCameraTrainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        *args, **kwargs):
    return CameraTrainerWrapper(
        lambda model, scene_extent, dataset, *args, **kwargs: DepthOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        model, dataset, scene_extent, *args, **kwargs
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


# Aliases for default trainers

Trainer = BaseDepthTrainer
CameraTrainer = DepthCameraTrainer
OpacityResetDensificationTrainer = DepthOpacityResetDensificationTrainer
OpacityResetDensificationCameraTrainer = DepthOpacityResetDensificationCameraTrainer
SHLiftTrainer = DepthSHLiftTrainer
SHLiftCameraTrainer = DepthSHLiftCameraTrainer
SHLiftOpacityResetDensificationTrainer = DepthSHLiftOpacityResetDensificationTrainer
SHLiftOpacityResetDensificationCameraTrainer = DepthSHLiftOpacityResetDensificationCameraTrainer
