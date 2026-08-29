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


def build_densifier(values: list[str], model: GaussianModel, dataset: CameraDataset, **configs) -> AbstractDensifier:
    """Wrap NoopDensifier with registry densifiers, applied inside-out.

    Each config key must belong to exactly one densifier in the list.
    """
    param_users: Dict[str, List[str]] = defaultdict(list)
    wraps: List[DensifierEntry] = []
    for wrap_name in values:
        entry = DENSIFIERS[wrap_name]
        for p in entry.params:
            param_users[p].append(wrap_name)
        wraps.append(entry)

    duplicated = {p: users for p, users in param_users.items() if p in configs and len(users) > 1}
    if duplicated:
        raise ValueError(f"duplicate params: {duplicated}")

    unused = [k for k in configs if k not in param_users]
    if unused:
        raise TypeError(f"unused input configs: {unused}; accepted: {sorted(param_users)}")

    split = {name: {} for name in values}
    for k, v in configs.items():
        split[param_users[k][0]][k] = v

    densifier = NoopDensifier(model, dataset)
    for wrap_name, entry in zip(values, wraps):
        densifier = entry.build(densifier, dataset, **split[wrap_name])
    return densifier


class DensifyTrainerEntry(TrainerRootEntry):
    def params_for(self, values: list[str]) -> Tuple[str, ...]:
        params = list(p.name for p in inspect.signature(BaseTrainer.__init__).parameters.values())[3:]
        for wrap_name in values:
            params.extend(DENSIFIERS[wrap_name].params)
        return tuple(params)

    def construct(self, values: list[str], model: GaussianModel, dataset: CameraDataset, **configs):
        trainer_keys = set(p.name for p in list(inspect.signature(BaseTrainer.__init__).parameters.values())[3:])
        trainer_configs = {k: v for k, v in configs.items() if k in trainer_keys}
        densifier_configs = {k: v for k, v in configs.items() if k not in trainer_keys}
        densifier = build_densifier(values, model, dataset, **densifier_configs)
        return self.cls(model, dataset, densifier, **trainer_configs)
