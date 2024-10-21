from gaussian_splatting import CameraTrainableGaussianModel
from gaussian_splatting.dataset import TrainableCameraDataset
from gaussian_splatting.utils import get_expon_lr_func
from .trainer import LRScheduledTrainer


class CameraTrainer(LRScheduledTrainer):
    def __init__(
            self,  model: CameraTrainableGaussianModel,
            dataset: TrainableCameraDataset,
            spatial_lr_scale: float,
            camera_position_lr_init=0.00016,
            camera_position_lr_final=0.0000016,
            camera_position_lr_delay_mult=0.01,
            camera_position_lr_max_steps=30_000,
            camera_rotation_lr_init=0.001,
            camera_rotation_lr_final=0.00001,
            camera_rotation_lr_delay_mult=0.01,
            camera_rotation_lr_max_steps=30_000,
            *args, **kwargs):
        super().__init__(
            model=model,
            spatial_lr_scale=spatial_lr_scale,
            *args, **kwargs
        )
        self.optimizer.add_param_group({'params': [dataset.quaternions], 'lr': camera_rotation_lr_init, "name": "quaternions"})
        self.quaternions_scheduler_args = get_expon_lr_func(
            lr_init=camera_rotation_lr_init,
            lr_final=camera_rotation_lr_final,
            lr_delay_mult=camera_rotation_lr_delay_mult,
            max_steps=camera_rotation_lr_max_steps,
        )
        self.optimizer.add_param_group({'params': [dataset.Ts], 'lr': camera_position_lr_init * spatial_lr_scale, "name": "Ts"})
        self.Ts_scheduler_args = get_expon_lr_func(
            lr_init=camera_position_lr_init*spatial_lr_scale,
            lr_final=camera_position_lr_final*spatial_lr_scale,
            lr_delay_mult=camera_position_lr_delay_mult,
            max_steps=camera_position_lr_max_steps,
        )

    def update_learning_rate(self, step: int):
        super().update_learning_rate(step)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "quaternions":
                lr = self.quaternions_scheduler_args(step)
                param_group['lr'] = lr
            if param_group["name"] == "Ts":
                lr = self.Ts_scheduler_args(step)
                param_group['lr'] = lr
