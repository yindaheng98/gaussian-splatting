from gaussian_splatting import CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.utils import get_expon_lr_func
from .abc import AbstractTrainer, TrainerWrapper
from .base import BaseTrainer


class CameraOptimizer(TrainerWrapper):
    def __init__(
            self,  base_trainer: AbstractTrainer,
            dataset: TrainableCameraDataset,
            spatial_lr_scale: float,
            camera_rotation_lr_init=0.0001,
            camera_rotation_lr_final=0.000001,
            camera_rotation_lr_delay_mult=0.01,
            camera_rotation_lr_max_steps=30_000,
            camera_position_lr_init=0.00016,
            camera_position_lr_final=0.0000016,
            camera_position_lr_delay_mult=0.01,
            camera_position_lr_max_steps=30_000,
            camera_exposure_lr_init=0.001,
            camera_exposure_lr_final=0.0001,
            camera_exposure_lr_delay_mult=0.01,
            camera_exposure_lr_max_steps=30_000):
        super().__init__(base_trainer)
        self.optimizer.add_param_group({'params': [dataset.quaternions], 'lr': camera_rotation_lr_init, "name": "quaternions"})
        self.optimizer.add_param_group({'params': [dataset.Ts], 'lr': camera_position_lr_init * spatial_lr_scale, "name": "Ts"})
        self.optimizer.add_param_group({'params': [dataset.exposures], 'lr': camera_exposure_lr_init, "name": "exposures"})
        self.schedulers["quaternions"] = get_expon_lr_func(
            lr_init=camera_rotation_lr_init,
            lr_final=camera_rotation_lr_final,
            lr_delay_mult=camera_rotation_lr_delay_mult,
            max_steps=camera_rotation_lr_max_steps,
        )
        self.schedulers["Ts"] = get_expon_lr_func(
            lr_init=camera_position_lr_init*spatial_lr_scale,
            lr_final=camera_position_lr_final*spatial_lr_scale,
            lr_delay_mult=camera_position_lr_delay_mult,
            max_steps=camera_position_lr_max_steps,
        )
        self.schedulers["exposures"] = get_expon_lr_func(
            lr_init=camera_exposure_lr_init,
            lr_final=camera_exposure_lr_final,
            lr_delay_mult=camera_exposure_lr_delay_mult,
            max_steps=camera_exposure_lr_max_steps,
        )


def BaseCameraTrainer(
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
        BaseTrainer(model, scene_extent, *args, **kwargs),
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
