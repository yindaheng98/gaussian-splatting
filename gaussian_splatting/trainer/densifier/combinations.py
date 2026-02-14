from functools import partial
from typing import Callable
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import AbstractDensifier, NoopDensifier
from .densifier import SplitCloneDensifierWrapper
from .pruner import OpacityPrunerDensifierWrapper
from .trainer import DensificationTrainer


def DensificationDensifierWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs) -> AbstractDensifier:
    return OpacityPrunerDensifierWrapper(
        partial(SplitCloneDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs
    )


def DensificationTrainerWrapper(
        base_densifier_constructor: Callable[..., AbstractDensifier],  # this is not Callable[..., AbstractTrainer]. Since DensificationTrainer cannot contain a base_trainer
        model: GaussianModel, dataset: CameraDataset, *args,
        **configs):
    return DensificationTrainer.from_densifier_constructor(
        partial(DensificationDensifierWrapper, base_densifier_constructor),
        model, dataset, *args,
        **configs
    )


def BaseDensificationTrainer(
        model: GaussianModel, dataset: CameraDataset,
        **configs):
    return DensificationTrainerWrapper(
        lambda model, dataset, **configs: NoopDensifier(model),
        model, dataset,
        **configs
    )
