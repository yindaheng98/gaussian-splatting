from functools import partial
from typing import Callable

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from ..abc import AbstractTrainer
from .depth_distortion import DepthDistortionTrainerWrapper
from .normal_consistency import NormalConsistencyTrainer, NormalConsistencyTrainerWrapper


def NormalTrainerWrapper(
        base_trainer_constructor: Callable[..., AbstractTrainer],
        model: GaussianModel,
        dataset: CameraDataset,
        *args,
        **configs) -> NormalConsistencyTrainer:
    return NormalConsistencyTrainerWrapper(
        partial(DepthDistortionTrainerWrapper, base_trainer_constructor),
        model,
        dataset,
        *args,
        **configs,
    )
