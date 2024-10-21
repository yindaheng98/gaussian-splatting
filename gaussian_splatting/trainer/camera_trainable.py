from gaussian_splatting import CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.utils import get_expon_lr_func
from .trainer import AbstractTrainer, TrainerWrapper
from .lr_scheduler import LRSchedulerWrapper, LRScheduledTrainer


class CameraOptimizer(TrainerWrapper):
    def __init__(
            self,  base_trainer: AbstractTrainer,
            dataset: TrainableCameraDataset,
            spatial_lr_scale: float,
            camera_position_lr=0.00016,
            camera_rotation_lr=0.0001):
        super().__init__(base_trainer)
        self.optimizer.add_param_group({'params': [dataset.quaternions], 'lr': camera_position_lr, "name": "quaternions"})
        self.optimizer.add_param_group({'params': [dataset.Ts], 'lr': camera_rotation_lr * spatial_lr_scale, "name": "Ts"})


class CameraLRScheduler(LRSchedulerWrapper):
    def __init__(
            self,  base_trainer: AbstractTrainer,
            spatial_lr_scale: float,
            camera_position_lr_init=0.00016,
            camera_position_lr_final=0.0000016,
            camera_position_lr_delay_mult=0.01,
            camera_position_lr_max_steps=30_000,
            camera_rotation_lr_init=0.0001,
            camera_rotation_lr_final=0.000001,
            camera_rotation_lr_delay_mult=0.01,
            camera_rotation_lr_max_steps=30_000):
        super().__init__(base_trainer)
        self.quaternions_scheduler_args = get_expon_lr_func(
            lr_init=camera_rotation_lr_init,
            lr_final=camera_rotation_lr_final,
            lr_delay_mult=camera_rotation_lr_delay_mult,
            max_steps=camera_rotation_lr_max_steps,
        )
        self.Ts_scheduler_args = get_expon_lr_func(
            lr_init=camera_position_lr_init*spatial_lr_scale,
            lr_final=camera_position_lr_final*spatial_lr_scale,
            lr_delay_mult=camera_position_lr_delay_mult,
            max_steps=camera_position_lr_max_steps,
        )

    def update_learning_rate(self, step: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "quaternions":
                lr = self.quaternions_scheduler_args(step)
                param_group['lr'] = lr
            if param_group["name"] == "Ts":
                lr = self.Ts_scheduler_args(step)
                param_group['lr'] = lr


def LRScheduledCameraTrainer(
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
    return CameraLRScheduler(
        CameraOptimizer(
            LRScheduledTrainer(model, scene_extent=scene_extent, *args, **kwargs),
            dataset=dataset, spatial_lr_scale=scene_extent,
            camera_position_lr=camera_position_lr_init,
            camera_rotation_lr=camera_rotation_lr_init
        ),
        spatial_lr_scale=scene_extent,
        camera_position_lr_init=camera_position_lr_init,
        camera_position_lr_final=camera_position_lr_final,
        camera_position_lr_delay_mult=camera_position_lr_delay_mult,
        camera_position_lr_max_steps=camera_position_lr_max_steps,
        camera_rotation_lr_init=camera_rotation_lr_init,
        camera_rotation_lr_final=camera_rotation_lr_final,
        camera_rotation_lr_delay_mult=camera_rotation_lr_delay_mult,
        camera_rotation_lr_max_steps=camera_rotation_lr_max_steps
    )
