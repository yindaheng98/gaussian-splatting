from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel

from .abc import AbstractDensifier, NoopDensifier
from .densifier import SplitCloneDensifierWrapper
from .pruner import OpacityPrunerDensifierWrapper
from .trainer import DensificationTrainer


def DensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float,
        *args, **kwargs) -> AbstractDensifier:
    return OpacityPrunerDensifierWrapper(
        partial(SplitCloneDensifierWrapper, base_densifier_constructor),
        model, scene_extent,
        *args, **kwargs
    )


def DensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float,
        *args, **kwargs):
    return DensificationTrainer.from_densifier_constructor(
        partial(DensifierWrapper, base_densifier_constructor),
        model, scene_extent,
        *args, **kwargs
    )


def BaseDensificationTrainer(
        model: GaussianModel, scene_extent: float,
        *args, **kwargs):
    return DensificationTrainerWrapper(
        lambda model, *args, **kwargs: NoopDensifier(model),
        model, scene_extent,
        *args, **kwargs
    )
