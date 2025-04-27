from typing import Callable
import torch
import torch.nn as nn

from gaussian_splatting.gaussian_model import GaussianModel

from .abc import AbstractTrainer, TrainerWrapper


def replace_tensor_to_optimizer(optimizer: torch.optim.Optimizer, tensor, name):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if group["name"] == name:
            stored_state = optimizer.state.get(group['params'][0], None)
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


class OpacityResetter(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            opacity_reset_from_iter=3000,
            opacity_reset_until_iter=15000,
            opacity_reset_interval=3000,
    ):
        super().__init__(base_trainer)
        self.opacity_reset_from_iter = opacity_reset_from_iter
        self.opacity_reset_until_iter = opacity_reset_until_iter
        self.opacity_reset_interval = opacity_reset_interval

    def optim_step(self):
        with torch.no_grad():
            if self.opacity_reset_from_iter <= self.curr_step <= self.opacity_reset_until_iter and self.curr_step % self.opacity_reset_interval == 0:
                opacities_new = self.model.inverse_opacity_activation(torch.min(self.model.get_opacity, torch.ones_like(self.model.get_opacity)*0.01))
                optimizable_tensors = replace_tensor_to_optimizer(self.optimizer, opacities_new, "opacity")
                self.model._opacity = optimizable_tensors["opacity"]
                torch.cuda.empty_cache()
        return super().optim_step()


def OpacityResetTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        scene_extent: float,
        *args,
        opacity_reset_from_iter=3000,
        opacity_reset_until_iter=15000,
        opacity_reset_interval=3000,
        **kwargs) -> OpacityResetter:
    return OpacityResetter(
        base_trainer=base_trainer_constructor(model, scene_extent, *args, **kwargs),
        opacity_reset_from_iter=opacity_reset_from_iter,
        opacity_reset_until_iter=opacity_reset_until_iter,
        opacity_reset_interval=opacity_reset_interval,
    )
