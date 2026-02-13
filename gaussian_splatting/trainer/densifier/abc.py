from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
import torch

from gaussian_splatting import GaussianModel


@dataclass(frozen=True)  # frozen=True just like NamedTuple
class DensificationInstruct:
    new_xyz: torch.Tensor = None
    new_features_dc: torch.Tensor = None
    new_features_rest: torch.Tensor = None
    new_opacity: torch.Tensor = None
    new_scaling: torch.Tensor = None
    new_rotation: torch.Tensor = None
    remove_mask: torch.Tensor = None
    replace_xyz_mask: torch.Tensor = None
    replace_xyz: torch.Tensor = None
    replace_features_dc_mask: torch.Tensor = None
    replace_features_dc: torch.Tensor = None
    replace_features_rest_mask: torch.Tensor = None
    replace_features_rest: torch.Tensor = None
    replace_opacity_mask: torch.Tensor = None
    replace_opacity: torch.Tensor = None
    replace_scaling_mask: torch.Tensor = None
    replace_scaling: torch.Tensor = None
    replace_rotation_mask: torch.Tensor = None
    replace_rotation: torch.Tensor = None

    # NamedTuple-compatible API

    def _asdict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def _replace(self, **kwargs):
        return type(self)(**{**self._asdict(), **kwargs})

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
                return b
            if b_mask is None:
                return a
            tmp = torch.zeros((a_mask.shape[0], *a.shape[1:]), device=a.device, dtype=a.dtype)
            tmp[a_mask, ...] = a
            tmp[b_mask, ...] = b
            mask = torch.logical_or(a_mask, b_mask)
            return tmp[mask, ...]

        merged_kwargs = {}
        for f in fields(a):
            name = f.name
            a_val = getattr(a, name)
            b_val = getattr(b, name)
            if name.startswith("new_"):
                merged_kwargs[name] = cat_new(a_val, b_val)
            elif name.endswith("_mask"):
                merged_kwargs[name] = or_mask(a_val, b_val)
            elif name.startswith("replace_"):
                mask_name = f"{name}_mask"
                merged_kwargs[name] = cover_replace(
                    getattr(a, mask_name), a_val,
                    getattr(b, mask_name), b_val
                )

        return type(a)(**merged_kwargs)


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
        return DensificationInstruct()
