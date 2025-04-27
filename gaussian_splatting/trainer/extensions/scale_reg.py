
import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.trainer import AbstractTrainer, TrainerWrapper, BaseTrainer


class ScaleRegularizer(TrainerWrapper):
    def __init__(
            self, base_trainer: AbstractTrainer,
            scene_extent: float,
            scale_reg_from_iter=3000,
            scale_reg_thr_scale=1.0,
            scale_reg_weight=0.1,
    ):
        super().__init__(base_trainer)
        self.scale_reg_scale = scene_extent
        self.scale_reg_from_iter = scale_reg_from_iter
        self.scale_reg_thr_scale = scale_reg_thr_scale
        self.scale_reg_weight = scale_reg_weight

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if self.curr_step < self.scale_reg_from_iter:
            return super().loss(out, camera)
        scale_reg_thr = self.scale_reg_thr_scale * self.scale_reg_scale
        scaling = self.model.get_scaling
        scaling_for_reg = scaling[scaling > scale_reg_thr]
        scale_reg = 0
        if scaling_for_reg.shape[0] > 0:
            scale_reg = (scaling[scaling > scale_reg_thr] / scale_reg_thr - 1).mean()
        return super().loss(out, camera) + scale_reg * self.scale_reg_weight


def ScaleRegularizeTrainerWrapper(
    base_constructor,
    model: GaussianModel,
    scene_extent: float,
    *args,
    scale_reg_from_iter=3000,
    scale_reg_thr_scale=1.0,
    scale_reg_weight=0.1,
    **kwargs
) -> ScaleRegularizer:
    return ScaleRegularizer(
        base_constructor(
            model,
            scene_extent,
            *args, **kwargs
        ),
        scene_extent=scene_extent,
        scale_reg_from_iter=scale_reg_from_iter,
        scale_reg_thr_scale=scale_reg_thr_scale,
        scale_reg_weight=scale_reg_weight,
    )


def BaseScaleRegularizeTrainer(model: GaussianModel, scene_extent: float, *args, **kwargs) -> ScaleRegularizer:
    return ScaleRegularizeTrainerWrapper(BaseTrainer, model, scene_extent, *args, **kwargs)
