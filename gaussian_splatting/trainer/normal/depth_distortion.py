from typing import Callable

import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from ..abc import AbstractTrainer, TrainerWrapper
from ..registry import trainer_wrap


class DepthDistortionTrainer(TrainerWrapper):

    def __init__(
            self,
            base_trainer: AbstractTrainer,
            depth_distortion_from_iter=3000,
            depth_distortion_weight=0.01,
    ):
        super().__init__(base_trainer)
        self.depth_distortion_from_iter = depth_distortion_from_iter
        self.depth_distortion_weight = depth_distortion_weight

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if "render_distort" not in out:
            raise KeyError(
                "DepthDistortionTrainer requires out['render_distort']; "
                "use a renderer with distortion output enabled"
            )

        loss = super().loss(out, camera)
        if self.curr_step <= self.depth_distortion_from_iter:
            return loss
        depth_distortion = out["render_distort"]
        return loss + depth_distortion.mean() * self.depth_distortion_weight


@trainer_wrap("depthdistortion")
def DepthDistortionTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        depth_distortion_from_iter=3000,
        depth_distortion_weight=0.01,
        **configs) -> DepthDistortionTrainer:
    return DepthDistortionTrainer(
        base_trainer=base_trainer_constructor(model, dataset, *args, **configs),
        depth_distortion_from_iter=depth_distortion_from_iter,
        depth_distortion_weight=depth_distortion_weight,
    )
