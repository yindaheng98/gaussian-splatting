from typing import Callable
from gaussian_splatting import GaussianModel

from .abc import AbstractDensifier, NoopDensifier
from .densifier import SplitCloneDensifier, SplitCloneDensifierTrainerWrapper
from .adaptive import AdaptiveSplitCloneDensifier, AdaptiveSplitCloneDensifierTrainerWrapper
from .pruner import OpacityPruner


def build_full_densifier(
    noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
    model: GaussianModel,
    scene_extent: float,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_interval=100,
    densify_grad_threshold=0.0002,
    densify_percent_dense=0.01,
    densify_percent_too_big=0.8,
    densify_limit_n=None,
    prune_from_iter=1000,
    prune_until_iter=15000,
    prune_interval=100,
    prune_screensize_threshold=20,
    prune_percent_too_big=1,
    prune_opacity_threshold=0.005,
) -> AbstractDensifier:
    base_densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = SplitCloneDensifier(
        base_densifier,
        scene_extent,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big,
        densify_limit_n=densify_limit_n
    )
    pruner = OpacityPruner(
        densifier,
        scene_extent,
        prune_from_iter=prune_from_iter,
        prune_until_iter=prune_until_iter,
        prune_interval=prune_interval,
        prune_screensize_threshold=prune_screensize_threshold,
        prune_percent_too_big=prune_percent_too_big,
        prune_opacity_threshold=prune_opacity_threshold
    )
    return pruner


def build_full_adaptive_densifier(
    noargs_base_densifier_constructor: Callable[[GaussianModel, float], AbstractDensifier],
    model: GaussianModel,
    scene_extent: float,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_interval=100,
    densify_grad_threshold=0.0002,
    densify_percent_dense=0.01,
    densify_percent_too_big=0.8,
    densify_limit_n=None,
    densify_target_n=10000,
    prune_from_iter=1000,
    prune_until_iter=15000,
    prune_interval=100,
    prune_screensize_threshold=20,
    prune_percent_too_big=1,
    prune_opacity_threshold=0.005,
) -> AbstractDensifier:
    base_densifier = noargs_base_densifier_constructor(model, scene_extent)
    densifier = AdaptiveSplitCloneDensifier(
        base_densifier,
        scene_extent,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        densify_grad_threshold=densify_grad_threshold,
        densify_percent_dense=densify_percent_dense,
        densify_percent_too_big=densify_percent_too_big,
        densify_limit_n=densify_limit_n,
        densify_target_n=densify_target_n
    )
    pruner = OpacityPruner(
        densifier,
        scene_extent,
        prune_from_iter=prune_from_iter,
        prune_until_iter=prune_until_iter,
        prune_interval=prune_interval,
        prune_screensize_threshold=prune_screensize_threshold,
        prune_percent_too_big=prune_percent_too_big,
        prune_opacity_threshold=prune_opacity_threshold
    )
    return pruner


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


def AdaptiveDensificationTrainerWrapper(
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
    return AdaptiveSplitCloneDensifierTrainerWrapper(
        lambda model, scene_extent: OpacityPruner(
            noargs_base_densifier_constructor(model, scene_extent),
            scene_extent,
            prune_from_iter=prune_from_iter,
            prune_until_iter=prune_until_iter,
            prune_interval=prune_interval,
            prune_screensize_threshold=prune_screensize_threshold,
            prune_percent_too_big=prune_percent_too_big,
            prune_opacity_threshold=prune_opacity_threshold,
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


def AdaptiveDensificationTrainer(
        model: GaussianModel,
        scene_extent: float,
        *args, **kwargs):
    return AdaptiveDensificationTrainerWrapper(
        lambda model, scene_extent: NoopDensifier(model),
        model,
        scene_extent,
        *args, **kwargs
    )
