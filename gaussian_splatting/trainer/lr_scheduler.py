

from abc import abstractmethod
from gaussian_splatting import GaussianModel
from gaussian_splatting.utils import get_expon_lr_func
from .trainer import AbstractTrainer, TrainerWrapper, BaseTrainer


class LRSchedulerWrapper(TrainerWrapper):
    def __init__(self, base_trainer: AbstractTrainer):
        super().__init__(base_trainer)
        self.curr_step = 1

    def optim_step(self):
        self.update_learning_rate(self.curr_step)
        super().optim_step()
        self.curr_step += 1

    @abstractmethod
    def update_learning_rate(self, step: int):
        pass


class BaseLRScheduler(LRSchedulerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            spatial_lr_scale: float,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=30_000
    ):
        super().__init__(base_trainer)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=position_lr_init*spatial_lr_scale,
            lr_final=position_lr_final*spatial_lr_scale,
            lr_delay_mult=position_lr_delay_mult,
            max_steps=position_lr_max_steps,
        )

    def update_learning_rate(self, step: int):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(step)
                param_group['lr'] = lr


def LRScheduledTrainer(
        model: GaussianModel,
        scene_extent: float,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        *args, **kwargs):
    return BaseLRScheduler(
        BaseTrainer(model, scene_extent, *args, **kwargs),
        scene_extent, position_lr_init, position_lr_final, position_lr_delay_mult, position_lr_max_steps
    )
