from typing import Callable

import torch

from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from ..abc import AbstractTrainer, TrainerWrapper


class NormalConsistencyTrainer(TrainerWrapper):

    def __init__(
            self,
            base_trainer: AbstractTrainer,
            normal_consistency_from_iter=7000,
            normal_consistency_weight=0.05,
    ):
        super().__init__(base_trainer)
        self.normal_consistency_from_iter = normal_consistency_from_iter
        self.normal_consistency_weight = normal_consistency_weight

    def loss(self, out: dict, camera: Camera) -> torch.Tensor:
        if not all(name in out for name in ("render_normals", "normals_from_depth", "render_alphas")):
            raise KeyError(
                "NormalConsistencyTrainer requires 2DGS normal outputs; "
                f"missing {[name for name in ('render_normals', 'normals_from_depth', 'render_alphas') if name not in out]}"
            )

        loss = super().loss(out, camera)
        if self.curr_step <= self.normal_consistency_from_iter:
            return loss

        render_normals = out["render_normals"]
        normals_from_depth = out["normals_from_depth"]
        render_alphas = out["render_alphas"]
        surface_normals = normals_from_depth * render_alphas.detach()
        normal_error = 1 - (render_normals * surface_normals).sum(dim=0)
        return loss + normal_error.mean() * self.normal_consistency_weight


def NormalConsistencyTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        normal_consistency_from_iter=7000,
        normal_consistency_weight=0.05,
        **configs) -> NormalConsistencyTrainer:
    return NormalConsistencyTrainer(
        base_trainer=base_trainer_constructor(model, dataset, *args, **configs),
        normal_consistency_from_iter=normal_consistency_from_iter,
        normal_consistency_weight=normal_consistency_weight,
    )
