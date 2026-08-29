import inspect
from abc import ABC
from collections.abc import Callable as AbcCallable
from functools import partial
from typing import Callable, Concatenate, Dict, Type, get_origin

from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset

from .abc import AbstractTrainer

TrainerConstructor = Callable[..., AbstractTrainer]
TrainerWrapFn = Callable[Concatenate[TrainerConstructor, GaussianModel, CameraDataset, ...], AbstractTrainer]

WRAPPER_SEP = "-"
ROOT_KEY_SEP = ":"
ROOT_VALUE_SEP = ","


class TrainerEntry(ABC):
    pass


class TrainerRootEntry(TrainerEntry):
    def __init__(self, cls: Type[AbstractTrainer]):
        if not issubclass(cls, AbstractTrainer):
            raise TypeError(f"{cls.__name__} must be a subclass of AbstractTrainer")
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
        self.cls = cls

    def construct(self, values: list[str], model: GaussianModel, dataset: CameraDataset, **configs) -> AbstractTrainer:
        return self.cls(model, dataset, **configs)


def validate_wrap_signature(fn: Callable):
    params = list(inspect.signature(fn).parameters.values())
    name = getattr(fn, "__name__", repr(fn))
    if len(params) < 5:
        raise TypeError(
            f"{name} must be (constructor, model, dataset, *args, ..., **configs), got {len(params)} parameters"
        )

    ctor, model, dataset, varargs, *rest = params
    if ctor.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) \
            or ctor.default is not inspect.Parameter.empty \
            or get_origin(ctor.annotation) is not AbcCallable:
        raise TypeError(
            f"{name} first parameter must be a Callable without a default, got {ctor.name}: {ctor.annotation}"
        )
    if model.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) \
            or model.default is not inspect.Parameter.empty \
            or not (isinstance(model.annotation, type) and issubclass(model.annotation, GaussianModel)):
        raise TypeError(
            f"{name} second parameter must be a GaussianModel subclass without a default, "
            f"got {model.name}: {model.annotation}"
        )
    if dataset.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) \
            or dataset.default is not inspect.Parameter.empty \
            or not (isinstance(dataset.annotation, type) and issubclass(dataset.annotation, CameraDataset)):
        raise TypeError(
            f"{name} third parameter must be a CameraDataset subclass without a default, "
            f"got {dataset.name}: {dataset.annotation}"
        )
    if varargs.kind is not inspect.Parameter.VAR_POSITIONAL:
        raise TypeError(f"{name} fourth parameter must be *args, got {varargs.name}")
    for p in rest[:-1]:
        if p.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(f"{name} parameters after *args must be keyword-only, got {p.name}")
        if p.default is inspect.Parameter.empty:
            raise TypeError(f"{name} parameter {p.name!r} after *args must have a default")
    if rest[-1].kind is not inspect.Parameter.VAR_KEYWORD:
        raise TypeError(f"{name} last parameter must be **configs, got {rest[-1].name}")


class TrainerWrapEntry(TrainerEntry):
    def __init__(self, fn: TrainerWrapFn):
        validate_wrap_signature(fn)
        self.fn = fn


TRAINERS: Dict[str, TrainerEntry] = {}
ALIASES: Dict[str, list[str]] = {}


def validate_key(key: str):
    if not isinstance(key, str):
        raise TypeError("key must be a string")
    if WRAPPER_SEP in key or ROOT_KEY_SEP in key or ROOT_VALUE_SEP in key:
        raise ValueError(f"key {key!r} must not contain {WRAPPER_SEP!r}, {ROOT_KEY_SEP!r} or {ROOT_VALUE_SEP!r}")
    if key in TRAINERS:
        raise ValueError(f"{key!r} is already registered as trainer: {TRAINERS[key]}")
    if key in ALIASES:
        raise ValueError(f"{key!r} is already registered as alias: {ALIASES[key]}")


def register(key: str, entry: TrainerEntry):
    validate_key(key)
    TRAINERS[key] = entry


def register_alias(key: str, keys: list[str]):
    validate_key(key)
    if not keys:
        raise ValueError("alias names must be a non-empty list")
    for k in keys:
        if k not in TRAINERS and k not in ALIASES:
            raise KeyError(f"{k!r} is not a registered trainer or alias")
    ALIASES[key] = list(keys)


def expand_alias(key: str) -> list[str]:
    if key not in ALIASES:
        return [key]
    return [n for part in ALIASES[key] for n in expand_alias(part)]


def parse_names(names: str) -> list[str]:
    return [n for name in names.split(WRAPPER_SEP) for n in expand_alias(name)]


def trainer_root(key: str, entry_cls: Type[TrainerRootEntry] = TrainerRootEntry):
    assert issubclass(entry_cls, TrainerRootEntry)

    def decorator(cls: Type[AbstractTrainer]) -> Type[AbstractTrainer]:
        register(key, entry_cls(cls))
        return cls
    return decorator


def trainer_wrap(key: str):
    def decorator(fn: TrainerWrapFn) -> TrainerWrapFn:
        register(key, TrainerWrapEntry(fn))
        return fn
    return decorator


def build_trainer(names: str, model: GaussianModel, dataset: CameraDataset, **configs) -> AbstractTrainer:
    """Construct a nested trainer from registry names.

    `names` is split by WRAPPER_SEP; each token that is a registered alias is expanded
    (recursively) in place. The first remaining token is a trainer_root, optionally
    key + ROOT_KEY_SEP + values joined by ROOT_VALUE_SEP; the rest are trainer_wraps
    applied inside-out. Each wrap peels its own kwargs and forwards **configs inward.
    """
    names = parse_names(names)
    if not names or not names[0]:
        raise ValueError("names must be a non-empty string")
    root_name, *wrap_names = names
    key, _, value = root_name.partition(ROOT_KEY_SEP)
    values = [t for t in value.split(ROOT_VALUE_SEP) if t]
    root = TRAINERS[key]
    if not isinstance(root, TrainerRootEntry):
        raise KeyError(f"first name {root_name!r} must be a trainer_root, got {sorted(n for n, e in TRAINERS.items() if isinstance(e, TrainerRootEntry))}")

    def constructor(model, dataset, **configs):
        return root.construct(values, model, dataset, **configs)

    for wrap_name in wrap_names:
        entry = TRAINERS[wrap_name]
        if not isinstance(entry, TrainerWrapEntry):
            raise KeyError(f"{wrap_name!r} must be a trainer_wrap, got {sorted(n for n, e in TRAINERS.items() if isinstance(e, TrainerWrapEntry))}")
        constructor = partial(entry.fn, constructor)
    return constructor(model, dataset, **configs)
