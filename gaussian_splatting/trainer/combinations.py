
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from .camera_trainable import CameraOptimizer, BaseCameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetter
from .sh_lift import SHLifter, BaseSHLiftTrainer
from .depth import DepthTrainer


# Camera trainer


def DepthCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_l1_weight_max_steps=30_000,
        *args, **kwargs):
    return DepthTrainer(
        BaseCameraTrainer(model, scene_extent, dataset, *args, **kwargs),
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps,
    )


# Densification trainers


def OpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,

        opacity_reset_from_iter=3000,
        opacity_reset_until_iter=15000,
        opacity_reset_interval=3000,

        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_l1_weight_max_steps=30_000,

        *args, **kwargs):
    base_trainer = OpacityResetter(
        BaseDensificationTrainer(
            model, scene_extent,
            *args, **kwargs
        ),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval
    )
    depth_trainer = DepthTrainer(
        base_trainer,
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps,
    )
    return depth_trainer


def OpacityResetDensificationCameraTrainer(
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
        OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
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


def DepthSHLiftTrainer(
        model: GaussianModel,
        scene_extent: float,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_l1_weight_max_steps=30_000,
        *args, **kwargs):
    return DepthTrainer(
        BaseSHLiftTrainer(model, scene_extent, *args, **kwargs),
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps,
    )


def SHLiftCameraTrainer(
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


def SHLiftOpacityResetDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        OpacityResetDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def SHLiftOpacityResetDensificationCameraTrainer(
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        OpacityResetDensificationCameraTrainer(model, scene_extent, dataset, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )
