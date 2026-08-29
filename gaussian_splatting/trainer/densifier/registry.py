import inspect
from collections import defaultdict
from typing import Dict, List, Tuple, Type

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from ..base import BaseTrainer
from ..registry import TrainerRootEntry, WRAPPER_SEP, ROOT_KEY_SEP, ROOT_VALUE_SEP
from .abc import AbstractDensifier, NoopDensifier


class DensifierEntry:
    def __init__(self, cls: Type[AbstractDensifier]):
        if not issubclass(cls, AbstractDensifier):
            raise TypeError(f"{cls.__name__} must be a subclass of AbstractDensifier")
        params = list(inspect.signature(cls.__init__).parameters.values())
        p = params[1]
        if p.default is not inspect.Parameter.empty or not issubclass(p.annotation, AbstractDensifier):
            raise TypeError(
                f"{cls.__name__}.__init__ first parameter must be an AbstractDensifier subclass "
                f"without a default, got {p.name}: {p.annotation}"
            )
        take_dataset = False
        if len(params) > 2:
            p = params[2]
            if p.default is inspect.Parameter.empty:
                if not issubclass(p.annotation, CameraDataset):
                    raise TypeError(
                        f"{cls.__name__}.__init__ second parameter, if required, must be a "
                        f"CameraDataset subclass without a default, got {p.name}: {p.annotation}"
                    )
                take_dataset = True
        self.cls = cls
        self.take_dataset = take_dataset

    @property
    def params(self) -> Tuple[str, ...]:
        return tuple(p.name for p in inspect.signature(self.cls.__init__).parameters.values())[3 if self.take_dataset else 2:]

    def build(self, densifier: AbstractDensifier, dataset: CameraDataset, **configs) -> AbstractDensifier:
        if self.take_dataset:
            return self.cls(densifier, dataset, **configs)
        return self.cls(densifier, **configs)


DENSIFIERS: Dict[str, DensifierEntry] = {}


def register(key: str, entry: DensifierEntry):
    if not isinstance(key, str):
        raise TypeError("densifier key must be a string")
    if WRAPPER_SEP in key or ROOT_KEY_SEP in key or ROOT_VALUE_SEP in key:
        raise ValueError(
            f"densifier key {key!r} must not contain {WRAPPER_SEP!r}, {ROOT_KEY_SEP!r} or {ROOT_VALUE_SEP!r}"
        )
    if key in DENSIFIERS:
        raise ValueError(f"densifier {key!r} is already registered: {DENSIFIERS[key]}")
    DENSIFIERS[key] = entry
    return entry.cls


def densifier(key: str):
    def decorator(cls: Type[AbstractDensifier]) -> Type[AbstractDensifier]:
        return register(key, DensifierEntry(cls))
    return decorator
