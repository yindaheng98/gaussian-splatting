from typing import Callable
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from .abc import AbstractTrainer
from .camera_trainable import CameraOptimizer, BaseCameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetter
from .sh_lift import SHLifter, BaseSHLiftTrainer
from .depth import DepthTrainer, DepthTrainerWrapper, BaseDepthTrainer


# Camera trainer


def DepthCameraTrainer(model: GaussianModel, scene_extent: float, dataset: TrainableCameraDataset, *args, **kwargs):
    return DepthTrainerWrapper(BaseCameraTrainer, model, scene_extent, dataset, *args, **kwargs)


# Densification trainers


def BaseOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        opacity_reset_from_iter=3000,
        opacity_reset_until_iter=15000,
        opacity_reset_interval=3000,
        *args, **kwargs):
    return OpacityResetter(
        BaseDensificationTrainer(
            model, scene_extent,
            *args, **kwargs
        ),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval
    )


def DepthOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        depth_from_iter=7500,
        depth_rescale_mode: str = 'local',
        depth_global_rescale_gt_max=10,
        depth_local_relative: bool = True,
        depth_local_relative_kernel_radius=8,
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
        depth_rescale_mode=depth_rescale_mode,
        depth_global_rescale_gt_max=depth_global_rescale_gt_max,
        depth_local_relative=depth_local_relative,
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
        camera_position_lr_init=0.00016,
        camera_position_lr_final=0.0000016,
        camera_position_lr_delay_mult=0.01,
        camera_position_lr_max_steps=30_000,
        camera_rotation_lr_init=0.0001,
        camera_rotation_lr_final=0.000001,
        camera_rotation_lr_delay_mult=0.01,
        camera_rotation_lr_max_steps=30_000,
        *args, **kwargs):
    return CameraOptimizer(
        BaseOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        dataset, scene_extent,
        camera_position_lr_init=camera_position_lr_init,
        camera_position_lr_final=camera_position_lr_final,
        camera_position_lr_delay_mult=camera_position_lr_delay_mult,
        camera_position_lr_max_steps=camera_position_lr_max_steps,
        camera_rotation_lr_init=camera_rotation_lr_init,
        camera_rotation_lr_final=camera_rotation_lr_final,
        camera_rotation_lr_delay_mult=camera_rotation_lr_delay_mult,
        camera_rotation_lr_max_steps=camera_rotation_lr_max_steps
    )


def DepthOpacityResetDensificationCameraTrainer(
        model: CameraTrainableGaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        camera_position_lr_init=0.00016,
        camera_position_lr_final=0.0000016,
        camera_position_lr_delay_mult=0.01,
        camera_position_lr_max_steps=30_000,
        camera_rotation_lr_init=0.0001,
        camera_rotation_lr_final=0.000001,
        camera_rotation_lr_delay_mult=0.01,
        camera_rotation_lr_max_steps=30_000,
        *args, **kwargs):
    return CameraOptimizer(
        DepthOpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        dataset, scene_extent,
        camera_position_lr_init=camera_position_lr_init,
        camera_position_lr_final=camera_position_lr_final,
        camera_position_lr_delay_mult=camera_position_lr_delay_mult,
        camera_position_lr_max_steps=camera_position_lr_max_steps,
        camera_rotation_lr_init=camera_rotation_lr_init,
        camera_rotation_lr_final=camera_rotation_lr_final,
        camera_rotation_lr_delay_mult=camera_rotation_lr_delay_mult,
        camera_rotation_lr_max_steps=camera_rotation_lr_max_steps
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
