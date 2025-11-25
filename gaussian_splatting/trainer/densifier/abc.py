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
    replace_xyz: torch.Tensor = None
    replace_features_dc: torch.Tensor = None
    replace_features_rest: torch.Tensor = None
    replace_opacities: torch.Tensor = None
    replace_scaling: torch.Tensor = None
    replace_rotation: torch.Tensor = None
    replace_mask: torch.Tensor = None

    @staticmethod
    def merge(a: 'DensificationInstruct', b: 'DensificationInstruct'):
        def cat_new(a: torch.Tensor, b: torch.Tensor):
            if a is None:
                return b
            if b is None:
                return a
            return torch.cat([a, b], dim=0)

        def or_mask(a: torch.Tensor, b: torch.Tensor):
            if a is None:
                return b
            if b is None:
                return a
            return torch.logical_or(a, b)

        def cover_replace(a_mask: torch.Tensor, a: torch.Tensor, b_mask: torch.Tensor, b: torch.Tensor):
            if a_mask is None:
                return b_mask, b
            if b_mask is None:
                return a_mask, a
            tmp = torch.zeros((a_mask.shape[0], *a.shape[1:]), device=a.device, dtype=a.dtype)
            tmp[a_mask] = a
            tmp[b_mask] = b
            mask = torch.logical_or(a_mask, b_mask)
            return tmp[mask, ...]

        return DensificationInstruct(
            new_xyz=cat_new(a.new_xyz, b.new_xyz),
            new_features_dc=cat_new(a.new_features_dc, b.new_features_dc),
            new_features_rest=cat_new(a.new_features_rest, b.new_features_rest),
            new_opacities=cat_new(a.new_opacities, b.new_opacities),
            new_scaling=cat_new(a.new_scaling, b.new_scaling),
            new_rotation=cat_new(a.new_rotation, b.new_rotation),
            remove_mask=or_mask(a.remove_mask, b.remove_mask),
            replace_mask=or_mask(a.replace_mask, b.replace_mask),
            replace_xyz=cover_replace(a.replace_mask, a.replace_xyz, b.replace_mask, b.replace_xyz),
            replace_features_dc=cover_replace(a.replace_mask, a.replace_features_dc, b.replace_mask, b.replace_features_dc),
            replace_features_rest=cover_replace(a.replace_mask, a.replace_features_rest, b.replace_mask, b.replace_features_rest),
            replace_opacities=cover_replace(a.replace_mask, a.replace_opacities, b.replace_mask, b.replace_opacities),
            replace_scaling=cover_replace(a.replace_mask, a.replace_scaling, b.replace_mask, b.replace_scaling),
            replace_rotation=cover_replace(a.replace_mask, a.replace_rotation, b.replace_mask, b.replace_rotation),
        )


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
