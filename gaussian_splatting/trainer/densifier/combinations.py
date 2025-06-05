from typing import Callable
from gaussian_splatting import GaussianModel

from .abc import AbstractDensifier, NoopDensifier
from .densifier import SplitCloneDensifierTrainerWrapper
from .pruner import OpacityPruner


def DensificationTrainerWrapper(
        noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
        model: GaussianModel,
        scene_extent: float,
        *args,
        prune_from_iter=1000,
        prune_until_iter=15000,
        prune_interval=100,
        prune_screensize_threshold=20,
        prune_percent_too_big=1,
        prune_opacity_threshold=0.005,
        **kwargs):
    return SplitCloneDensifierTrainerWrapper(
        lambda model, scene_extent: OpacityPruner(
            noargs_base_densifier_constructor(model, scene_extent),
            scene_extent,
            prune_from_iter=prune_from_iter,
            prune_until_iter=prune_until_iter,
            prune_interval=prune_interval,
            prune_screensize_threshold=prune_screensize_threshold,
            prune_percent_too_big=prune_percent_too_big,
            prune_opacity_threshold=prune_opacity_threshold
        ),
        model, scene_extent,
        *args, **kwargs
    )


def BaseDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        *args, **kwargs):
    return DensificationTrainerWrapper(
        lambda model, scene_extent: NoopDensifier(model),
        model,
        scene_extent,
        *args, **kwargs
    )
