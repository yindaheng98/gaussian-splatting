import inspect
from abc import ABC
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Type

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import AbstractTrainer


class TrainerEntry(ABC):
    def __init__(self, cls: Type[AbstractTrainer]):
        if not issubclass(cls, AbstractTrainer):
            raise TypeError(f"{cls.__name__} must be a subclass of AbstractTrainer")
        self.cls = cls


class TrainerRootEntry(TrainerEntry):
    def __init__(self, cls: Type[AbstractTrainer]):
        super().__init__(cls)
        params = list(inspect.signature(cls.__init__).parameters.values())
        p = params[1]
        if p.default is not inspect.Parameter.empty or not issubclass(p.annotation, GaussianModel):
            raise TypeError(
                f"{cls.__name__}.__init__ first parameter must be a GaussianModel subclass "
                f"without a default, got {p.name}: {p.annotation}"
            )
        p = params[2]
        if p.default is not inspect.Parameter.empty or not issubclass(p.annotation, CameraDataset):
            raise TypeError(
                f"{cls.__name__}.__init__ second parameter must be a CameraDataset subclass "
                f"without a default, got {p.name}: {p.annotation}"
            )

    def params_for(self, value: str) -> Tuple[str, ...]:
        return tuple(p.name for p in inspect.signature(self.cls.__init__).parameters.values())[3:]

    def construct(self, value: str, model: GaussianModel, dataset: CameraDataset, **configs) -> AbstractTrainer:
        return self.cls(model, dataset, **configs)


class TrainerWrapEntry(TrainerEntry):
    def __init__(self, cls: Type[AbstractTrainer]):
        super().__init__(cls)
        params = list(inspect.signature(cls.__init__).parameters.values())
        p = params[1]
        if p.default is not inspect.Parameter.empty or not issubclass(p.annotation, AbstractTrainer):
            raise TypeError(
                f"{cls.__name__}.__init__ first parameter must be an AbstractTrainer subclass "
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
        self.take_dataset = take_dataset

    def params_for(self) -> Tuple[str, ...]:
        return tuple(p.name for p in inspect.signature(self.cls.__init__).parameters.values())[3 if self.take_dataset else 2:]

    def build(self, trainer: AbstractTrainer, dataset: CameraDataset, **configs) -> AbstractTrainer:
        if self.take_dataset:
            return self.cls(trainer, dataset, **configs)
        return self.cls(trainer, **configs)


TRAINERS: Dict[str, TrainerEntry] = {}


def register(key: str, entry: TrainerEntry):
    if not isinstance(key, str):
        raise TypeError("trainer key must be a string")
    if key in TRAINERS:
        raise ValueError(f"trainer {key!r} is already registered: {TRAINERS[key]}")
    TRAINERS[key] = entry
    return entry.cls


def trainer(key: str, entry_cls: Type[TrainerEntry]):
    assert issubclass(entry_cls, TrainerEntry)

    def decorator(cls: Type[AbstractTrainer]) -> Type[AbstractTrainer]:
        return register(key, entry_cls(cls))
    return decorator


def trainer_root(key: str):
    return trainer(key, TrainerRootEntry)


def trainer_wrap(key: str):
    return trainer(key, TrainerWrapEntry)
