import inspect
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Tuple, Type

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import AbstractTrainer

WRAPPER_SEP = "-"
ROOT_KEY_SEP = ":"
ROOT_VALUE_SEP = "/"


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

    def params_for(self, values: list[str]) -> Tuple[str, ...]:
        return tuple(p.name for p in inspect.signature(self.cls.__init__).parameters.values())[3:]

    def construct(self, values: list[str], model: GaussianModel, dataset: CameraDataset, **configs) -> AbstractTrainer:
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

    @property
    def params(self) -> Tuple[str, ...]:
        return tuple(p.name for p in inspect.signature(self.cls.__init__).parameters.values())[3 if self.take_dataset else 2:]

    def build(self, trainer: AbstractTrainer, dataset: CameraDataset, **configs) -> AbstractTrainer:
        if self.take_dataset:
            return self.cls(trainer, dataset, **configs)
        return self.cls(trainer, **configs)


TRAINERS: Dict[str, TrainerEntry] = {}


def register(key: str, entry: TrainerEntry):
    if not isinstance(key, str):
        raise TypeError("trainer key must be a string")
    if WRAPPER_SEP in key or ROOT_KEY_SEP in key or ROOT_VALUE_SEP in key:
        raise ValueError(f"trainer key {key!r} must not contain {WRAPPER_SEP!r}, {ROOT_KEY_SEP!r} or {ROOT_VALUE_SEP!r}")
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


def build_trainer(names: str, model: GaussianModel, dataset: CameraDataset, **configs) -> AbstractTrainer:
    """Construct a nested trainer from registry names.

    `names` is split by NAME_SEP; the first token is a trainer_root, optionally key + VALUE_SEP + values
    joined by VALUES_SEP; the rest are trainer_wraps applied inside-out.
    Each config key must belong to exactly one trainer in the list.
    """
    names = names.split(WRAPPER_SEP)
    if not names or not names[0]:
        raise ValueError("names must be a non-empty string")
    root_name, *wrap_names = names
    key, _, value = root_name.partition(ROOT_KEY_SEP)
    values = [t for t in value.split(ROOT_VALUE_SEP) if t]
    root = TRAINERS[key]
    if not isinstance(root, TrainerRootEntry):
        raise KeyError(f"first name {root_name!r} must be a trainer_root, got {sorted(n for n, e in TRAINERS.items() if isinstance(e, TrainerRootEntry))}")
    param_users: Dict[str, List[str]] = defaultdict(list)
    for p in root.params_for(values):
        param_users[p].append(root_name)

    wraps: List[TrainerWrapEntry] = []
    for wrap_name in wrap_names:
        entry = TRAINERS[wrap_name]
        if not isinstance(entry, TrainerWrapEntry):
            raise KeyError(f"{wrap_name!r} must be a trainer_wrap, got {sorted(n for n, e in TRAINERS.items() if isinstance(e, TrainerWrapEntry))}")
        for p in entry.params:
            param_users[p].append(wrap_name)
        wraps.append(entry)

    duplicated = {p: users for p, users in param_users.items() if p in configs and len(users) > 1}
    if duplicated:
        raise ValueError(f"duplicate params: {duplicated}")

    unused = [k for k in configs if k not in param_users]
    if unused:
        raise TypeError(f"unused input configs: {unused}; accepted: {sorted(param_users)}")

    split = {name: {} for name in names}
    for k, v in configs.items():
        split[param_users[k][0]][k] = v

    trainer = root.construct(values, model, dataset, **split[root_name])
    for wrap_name, entry in zip(wrap_names, wraps):
        trainer = entry.build(trainer, dataset, **split[wrap_name])
    return trainer
