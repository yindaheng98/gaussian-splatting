from abc import ABC, abstractmethod
from typing import NamedTuple
import torch

from gaussian_splatting import GaussianModel


class DensificationInstruct(NamedTuple):
    new_xyz: torch.Tensor = None
    new_features_dc: torch.Tensor = None
    new_features_rest: torch.Tensor = None
    new_opacities: torch.Tensor = None
    new_scaling: torch.Tensor = None
    new_rotation: torch.Tensor = None
    remove_mask: torch.Tensor = None


class AbstractDensifier(ABC):

    @property
    @abstractmethod
    def model(self) -> GaussianModel:
        raise ValueError("Model is not set")

    @abstractmethod
    def densify_and_prune(self, loss, out, camera, step: int) -> DensificationInstruct:
        raise NotImplementedError
