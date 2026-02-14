from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel

from .abc import AbstractDensifier, NoopDensifier
from .densifier import SplitCloneDensifierWrapper
from .pruner import OpacityPrunerDensifierWrapper
from .trainer import DensificationTrainer


def DensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, scene_extent: float, *args,
        **configs) -> AbstractDensifier:
    return OpacityPrunerDensifierWrapper(
        partial(SplitCloneDensifierWrapper, base_densifier_constructor),
        model, scene_extent, *args,
        **configs
    )


def DensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],  # this is not Callable[..., AbstractTrainer]. Since DensificationTrainer cannot contain a base_trainer
        model: GaussianModel, scene_extent: float, *args,
        **configs):
    return DensificationTrainer.from_densifier_constructor(
        partial(DensificationDensifierWrapper, base_densifier_constructor),
        model, scene_extent, *args,
        **configs
    )


def BaseDensificationTrainer(
        model: GaussianModel, scene_extent: float, *args,
        **configs):
    return DensificationTrainerWrapper(
        lambda model, *args, **configs: NoopDensifier(model),
        model, scene_extent, *args,
        **configs
    )
