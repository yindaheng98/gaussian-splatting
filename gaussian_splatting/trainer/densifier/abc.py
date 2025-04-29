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

    def after_densify_and_prune_hook(self, loss, out, camera):
        pass


class DensifierWrapper(AbstractDensifier):
    '''
    This class is designed to wrap a trainer and add additional functionality.
    Without this class, you should modify the trainer class directly.
    '''

    def __init__(self, base_densifier: AbstractDensifier):
        super().__init__()
        self.base_densifier = base_densifier

    @property
    def model(self) -> GaussianModel:
        return self.base_densifier.model

    def densify_and_prune(self, loss, out, camera, step: int) -> DensificationInstruct:
        return self.base_densifier.densify_and_prune(loss, out, camera, step)

    def after_densify_and_prune_hook(self, loss, out, camera):
        return self.base_densifier.after_densify_and_prune_hook(loss, out, camera)


class NoopDensifier(AbstractDensifier):
    '''
    This class is designed to do nothing.
    It is used as the base of all densifier wrapper.
    '''

    def __init__(self, model: GaussianModel):
        super().__init__()
        self._model = model

    @property
    def model(self) -> GaussianModel:
        return self._model

    def densify_and_prune(self, loss, out, camera, step: int) -> DensificationInstruct:
        return DensificationInstruct(
            new_xyz=None,
            new_features_dc=None,
            new_features_rest=None,
            new_opacities=None,
            new_scaling=None,
            new_rotation=None,
            remove_mask=None
        )
