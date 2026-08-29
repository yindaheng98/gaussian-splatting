import inspect
from typing import Dict, Type

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import AbstractTrainer, TrainerWrapper

TRAINER_ROOTS: Dict[str, Type[AbstractTrainer]] = {}
TRAINER_WRAPS: Dict[str, Type[TrainerWrapper]] = {}
TRAINER_WRAPS_TAKES_DATASET: Dict[str, bool] = {}


def trainer_root(name: str):
    """Register a direct subclass of AbstractTrainer (the root of a wrapping chain)."""
    if not isinstance(name, str):
        raise TypeError("@trainer_root requires a name string, e.g. @trainer_root('base')")

    def decorator(cls: Type[AbstractTrainer]) -> Type[AbstractTrainer]:
        if AbstractTrainer not in cls.__bases__:
            raise TypeError(f"@trainer_root is for direct subclasses of AbstractTrainer, got {cls.__name__}")
        if issubclass(cls, TrainerWrapper):
            raise TypeError(f"{cls.__name__} is a TrainerWrapper; use @trainer_wrap instead")

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

        if name in TRAINER_ROOTS:
            raise ValueError(f"trainer_root {name!r} is already registered: {TRAINER_ROOTS[name]}")
        TRAINER_ROOTS[name] = cls
        return cls

    return decorator


def trainer_wrap(name: str):
    """Register a subclass of TrainerWrapper."""
    if not isinstance(name, str):
        raise TypeError("@trainer_wrap requires a name string, e.g. @trainer_wrap('depth')")

    def decorator(cls: Type[TrainerWrapper]) -> Type[TrainerWrapper]:
        if not issubclass(cls, TrainerWrapper) or cls is TrainerWrapper:
            raise TypeError(
                f"@trainer_wrap is for subclasses of TrainerWrapper, got {cls.__name__}"
            )

        params = list(inspect.signature(cls.__init__).parameters.values())

        p = params[1]
        if p.default is not inspect.Parameter.empty or not issubclass(p.annotation, AbstractTrainer):
            raise TypeError(
                f"{cls.__name__}.__init__ first parameter must be an AbstractTrainer subclass "
                f"without a default, got {p.name}: {p.annotation}"
            )

        takes_dataset = False
        if len(params) > 2:
            p = params[2]
            if p.default is inspect.Parameter.empty:
                if not issubclass(p.annotation, CameraDataset):
                    raise TypeError(
                        f"{cls.__name__}.__init__ second parameter, if required, must be a "
                        f"CameraDataset subclass without a default, got {p.name}: {p.annotation}"
                    )
                takes_dataset = True

        if name in TRAINER_WRAPS:
            raise ValueError(f"trainer_wrap {name!r} is already registered: {TRAINER_WRAPS[name]}")
        TRAINER_WRAPS[name] = cls
        TRAINER_WRAPS_TAKES_DATASET[name] = takes_dataset
        return cls

    return decorator
