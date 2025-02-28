from abc import ABC, abstractmethod
from typing import NamedTuple
import torch


class DensificationInstruct(NamedTuple):
    new_xyz: torch.Tensor = None
    new_features_dc: torch.Tensor = None
    new_features_rest: torch.Tensor = None
    new_opacities: torch.Tensor = None
    new_scaling: torch.Tensor = None
    new_rotation: torch.Tensor = None
    remove_mask: torch.Tensor = None


class AbstractDensifier(ABC):

    @abstractmethod
    def densify_and_prune(self, loss, out, camera, step: int) -> DensificationInstruct:
        raise NotImplementedError
