
from gaussian_splatting import GaussianModel, CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from .camera_trainable import CameraOptimizer, BaseCameraTrainer
from .densifier import BaseDensificationTrainer
from .opacity_reset import OpacityResetter
from .sh_lift import SHLifter, BaseSHLiftTrainer
from .depth import DepthTrainer


def DepthTrainerWrapper(
    base_constructor,
        model: GaussianModel,
        scene_extent: float,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_l1_weight_max_steps=30_000,
        *args, **kwargs):
    return DepthTrainer(
        base_constructor(model, scene_extent, *args, **kwargs),
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps,
    )


def DepthCameraTrainerWrapper(
    base_constructor,
        model: GaussianModel,
        scene_extent: float,
        dataset: TrainableCameraDataset,
        depth_l1_weight_init=1.0,
        depth_l1_weight_final=0.01,
        depth_l1_weight_max_steps=30_000,
        *args, **kwargs):
    return DepthTrainer(
        base_constructor(model, scene_extent, dataset, *args, **kwargs),
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        depth_l1_weight_max_steps=depth_l1_weight_max_steps,
    )


def DepthCameraTrainer(*args, **kwargs):
    return DepthCameraTrainerWrapper(BaseCameraTrainer, *args, **kwargs)


def OpacityResetDensificationTrainer(
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


def DepthOpacityResetDensificationTrainer(*args, **kwargs):
    return DepthTrainerWrapper(OpacityResetDensificationTrainer, *args, **kwargs)


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


def DepthOpacityResetDensificationCameraTrainer(*args, **kwargs):
    return DepthCameraTrainerWrapper(OpacityResetDensificationCameraTrainer, *args, **kwargs)


def DepthSHLiftTrainer(*args, **kwargs):
    return DepthTrainerWrapper(BaseSHLiftTrainer, *args, **kwargs)


def SHLiftCameraTrainer(
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


def DepthSHLiftCameraTrainer(*args, **kwargs):
    return DepthCameraTrainerWrapper(SHLiftCameraTrainer, *args, **kwargs)


def SHLiftDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        sh_degree_up_interval=1000,
        initial_sh_degree=0,
        *args, **kwargs):
    return SHLifter(
        BaseDensificationTrainer(model, scene_extent, *args, **kwargs),
        sh_degree_up_interval=sh_degree_up_interval,
        initial_sh_degree=initial_sh_degree
    )


def DepthSHLiftDensificationTrainer(*args, **kwargs):
    return DepthTrainerWrapper(SHLiftDensificationTrainer, *args, **kwargs)


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


def DepthSHLiftOpacityResetDensificationTrainer(*args, **kwargs):
    return DepthTrainerWrapper(SHLiftOpacityResetDensificationTrainer, *args, **kwargs)


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


def DepthSHLiftOpacityResetDensificationCameraTrainer(*args, **kwargs):
    return DepthCameraTrainerWrapper(SHLiftOpacityResetDensificationCameraTrainer, *args, **kwargs)
