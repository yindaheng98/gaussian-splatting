import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Type

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import AbstractTrainer


class TrainerEntry(ABC):
    def __init__(self, name: str, cls: Type[AbstractTrainer]):
        if not isinstance(name, str):
            raise TypeError("trainer name must be a string")
        if not issubclass(cls, AbstractTrainer):
            raise TypeError(f"{cls.__name__} must be a subclass of AbstractTrainer")
        self.name = name
        self.cls = cls

    @abstractmethod
    def params_for(self, name: str) -> Tuple[str, ...]:
        ...


class TrainerRootEntry(TrainerEntry):
    def __init__(self, name: str, cls: Type[AbstractTrainer]):
        super().__init__(name, cls)
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

    def params_for(self, name: str) -> Tuple[str, ...]:
        return tuple(p.name for p in inspect.signature(self.cls.__init__).parameters.values())[3:]

    def construct(self, name: str, model: GaussianModel, dataset: CameraDataset, **configs) -> AbstractTrainer:
        return self.cls(model, dataset, **configs)


class TrainerWrapEntry(TrainerEntry):
    def __init__(self, name: str, cls: Type[AbstractTrainer]):
        super().__init__(name, cls)
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

    def params_for(self, name: str) -> Tuple[str, ...]:
        return tuple(p.name for p in inspect.signature(self.cls.__init__).parameters.values())[3 if self.take_dataset else 2:]

    def construct(self, name: str, trainer: AbstractTrainer, dataset: CameraDataset, **configs) -> AbstractTrainer:
        if self.take_dataset:
            return self.cls(trainer, dataset, **configs)
        return self.cls(trainer, **configs)


TRAINERS: Dict[str, TrainerEntry] = {}


def register(entry: TrainerEntry):
    if entry.name in TRAINERS:
        raise ValueError(f"trainer {entry.name!r} is already registered: {TRAINERS[entry.name]}")
    TRAINERS[entry.name] = entry
    return entry.cls


def trainer(name: str, entry_cls: Type[TrainerEntry]):
    assert issubclass(entry_cls, TrainerEntry)

    def decorator(cls: Type[AbstractTrainer]) -> Type[AbstractTrainer]:
        return register(entry_cls(name, cls))
    return decorator


def trainer_root(name: str):
    return trainer(name, TrainerRootEntry)


def trainer_wrap(name: str):
    return trainer(name, TrainerWrapEntry)
