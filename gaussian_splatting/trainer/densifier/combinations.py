from typing import Callable
from gaussian_splatting import GaussianModel

from .abc import AbstractDensifier, NoopDensifier
from .densifier import Densifier
from .pruner import OpacityPruner
from .trainer import DensificationTrainer


def DensificationTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_interval=100,
        densify_grad_threshold=0.0002,
        densify_percent_dense=0.01,
        densify_percent_too_big=0.8,
        *args, **kwargs):
    densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = Densifier(
        densifier,
        model, scene_extent,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big
    )
    return DensificationTrainer(
        model, scene_extent,
        densifier,
        *args, **kwargs
    )


def BaseDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval=100,
        prune_screensize_threshold=20,
        prune_percent_too_big=1,
        prune_opacity_threshold=0.005,
        *args, **kwargs):
    return DensificationTrainerWrapper(
        lambda model, scene_extent: OpacityPruner(
            NoopDensifier(model),
            model, scene_extent,
            prune_from_iter=prune_from_iter,
            prune_until_iter=prune_until_iter,
            prune_interval=prune_interval,
            prune_screensize_threshold=prune_screensize_threshold,
            prune_percent_too_big=prune_percent_too_big,
            prune_opacity_threshold=prune_opacity_threshold
        ),
        model,
        scene_extent,
        *args, **kwargs
    )
